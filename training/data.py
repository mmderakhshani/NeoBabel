# This work is based in part on code from Show-o (https://github.com/showlab/Show-o).
# Significant modifications for NeoBabel.

import itertools
import json
import math
import os
import random
import re
from functools import partial
from typing import List, Optional, Union, Dict, Any
from pathlib import Path 
from tqdm import tqdm
import csv

from PIL import Image

Image.warnings.simplefilter('error', Image.DecompressionBombWarning)

import webdataset as wds
import yaml
from braceexpand import braceexpand
from torch.utils.data import default_collate
from torchvision import transforms
from transformers import PreTrainedTokenizer
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)
from multiprocessing import Manager, Lock
import mmap
from multiprocessing.shared_memory import SharedMemory
import omegaconf

person_token = ["a person", "someone", "somebody"]

# Add constants at module level
SUPPORTED_LANGUAGES = [
    "English",
    "Western Persian", 
    "Dutch",
    "French",
    "Hindi",
    "Chinese (Simplified)"
]

def replace_person_token(t):
    "Used for CC12M"
    t = re.sub("<person>([,\s]*(and)*[,\s]*<person>)+", " people ", t)
    while "<person>" in t:
        t = t.replace("<person>", f" {random.choice(person_token)} ", 1)
    return t


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=wds.warn_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def image_transform(sample, resolution=256):
    image = sample["images"]
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
    sample["images"] = image
    return sample


def remove_prefix(caption):
    caption = caption.replace('The image features ', '').replace('The image presents ', '').replace(
        "The image you've sent is, ", '').replace("In the center of the image, ", '').replace(
        "The image showcases ", '').replace("The image is ", '').replace(
        "The image captures ", '').replace("In the given image ", '').replace(
        "The image portrays ", '').replace("In the image, ", '').replace("In this image, we see ", '').replace(
        "The image depicts ", '').replace("This is ", '').replace("In this image, ", '').replace(
        "This image captures ", '')

    return caption




class SharedJSONCache:
    """Uses shared memory to efficiently store JSON files across processes."""
    
    def __init__(self, name, data=None):
        self.json_bytes = json.dumps(data).encode() if data is not None else None
        try:
            # Try to create new shared memory
            self.shm = SharedMemory(name=name, create=True, size=len(self.json_bytes) if self.json_bytes else 0)
            # If we created it, write the data
            if self.json_bytes:
                self.shm.buf[:] = self.json_bytes
        except FileExistsError:
            # If it already exists, just attach to it
            self.shm = SharedMemory(name=name)
            
        self.size = len(self.json_bytes) if self.json_bytes else 0

    @classmethod
    def attach(cls, name):
        obj = cls.__new__(cls)
        obj.shm = SharedMemory(name=name)
        obj.size = len(obj.shm.buf)  # Add size attribute when attaching
        return obj

    def get_data(self):
        return json.loads(bytes(self.shm.buf[:self.size]))  # Deserialize when needed

    def close(self):
        self.shm.close()
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass


class Text2ImageDataset:
    def __init__(
            self,
            train_shards_path_or_url: Union[str, List[str], Dict[str, List[str]]],
            tokenizer: PreTrainedTokenizer,
            max_seq_length: int,
            num_train_examples: int,
            per_gpu_batch_size: int,
            global_batch_size: int,
            num_workers: int,
            resolution: int = 256,
            shuffle_buffer_size: int = 1000,
            pin_memory: bool = False,
            persistent_workers: bool = False,
            external_caption_path: Optional[str] = '',
            external_journeydb_caption_path: Optional[str] = '',
            external_laion12m_caption_path: Optional[str] = '',
            external_cc12m_caption_path: Optional[str] = '',
            external_crs3600_caption_path: Optional[str] = '',
            external_liu4k_gcp_caption_path: Optional[str] = '',
            external_open_img_pref_gcp_caption_path: Optional[str] = '',
            external_open_pickapic_gcp_caption_path: Optional[str] = '',
            external_instruction_tuning_NeoBabel_geneval_train_caption_path: Optional[str] = '',
            external_instruction_tuning_NeoBabel_human_gestures_caption_path: Optional[str] = '',
            external_instruction_tuning_NeoBabel_journey_caption_path: Optional[str] = '',
            external_instruction_tuning_NeoBabel_mscoco_human_caption_path: Optional[str] = '',
            external_instruction_tuning_NeoBabel_dalle3_caption_path: Optional[str] = '',
            external_instruction_tuning_NeoBabel_object_1_caption_path: Optional[str] = '',
            external_instruction_tuning_NeoBabel_object_2_caption_path: Optional[str] = '',
            external_instruction_tuning_NeoBabel_occupation_1_caption_path: Optional[str] = '',
            external_instruction_tuning_NeoBabel_occupation_2_caption_path: Optional[str] = '',
            external_instruction_tuning_NeoBabel_text_1_caption_path: Optional[str] = '',
            external_instruction_tuning_NeoBabel_text_2_caption_path: Optional[str] = '',
            is_captioning: bool = False,
            add_caption_prompt: bool = False,
            long_caption: bool = True,
            language: str = 'random', 
            shard_ratios: Optional[Dict[str, float]] = None,
    ):
        if f"{train_shards_path_or_url}.yaml" in os.listdir('./configs'):
            with open(f"./configs/{train_shards_path_or_url}.yaml") as f:
                train_shards_path_or_url = yaml.safe_load(f)
        self.long_caption = long_caption
        self.external_caption_path = external_caption_path
        self.external_journeydb_caption_path = external_journeydb_caption_path
        self.external_laion12m_caption_path = external_laion12m_caption_path
        self.external_cc12m_caption_path = external_cc12m_caption_path
        self.external_crs3600_caption_path = external_crs3600_caption_path
        self.external_liu4k_gcp_caption_path = external_liu4k_gcp_caption_path
        self.external_open_img_pref_gcp_caption_path = external_open_img_pref_gcp_caption_path
        self.external_open_pickapic_gcp_caption_path = external_open_pickapic_gcp_caption_path
        self.external_instruction_tuning_NeoBabel_geneval_train_caption_path = external_instruction_tuning_NeoBabel_geneval_train_caption_path
        self.external_instruction_tuning_NeoBabel_human_gestures_caption_path = external_instruction_tuning_NeoBabel_human_gestures_caption_path
        self.external_instruction_tuning_NeoBabel_journey_caption_path = external_instruction_tuning_NeoBabel_journey_caption_path
        self.external_instruction_tuning_NeoBabel_mscoco_human_caption_path = external_instruction_tuning_NeoBabel_mscoco_human_caption_path
        self.external_instruction_tuning_NeoBabel_dalle3_caption_path = external_instruction_tuning_NeoBabel_dalle3_caption_path
        self.external_instruction_tuning_NeoBabel_object_1_caption_path = external_instruction_tuning_NeoBabel_object_1_caption_path
        self.external_instruction_tuning_NeoBabel_object_2_caption_path = external_instruction_tuning_NeoBabel_object_2_caption_path
        self.external_instruction_tuning_NeoBabel_occupation_1_caption_path = external_instruction_tuning_NeoBabel_occupation_1_caption_path
        self.external_instruction_tuning_NeoBabel_occupation_2_caption_path = external_instruction_tuning_NeoBabel_occupation_2_caption_path
        self.external_instruction_tuning_NeoBabel_text_1_caption_path = external_instruction_tuning_NeoBabel_text_1_caption_path
        self.external_instruction_tuning_NeoBabel_text_2_caption_path = external_instruction_tuning_NeoBabel_text_2_caption_path
        self.is_captioning = is_captioning
        self.language = language  
        self.translations_cache = {}
        self.translations_cache_journeydb = {}
        self.translations_cache_crs3600 = {}
        self.translations_cache_liu4k_gcp = {}
        self.translations_cache_open_img_pref_gcp = {}
        self.translations_cache_open_pickapic_gcp = {}
        self.translations_cache_instruction_tuning_NeoBabel_geneval_train = {}
        self.translations_cache_instruction_tuning_NeoBabel_human_gestures = {}
        self.translations_cache_instruction_tuning_NeoBabel_journey = {}
        self.translations_cache_instruction_tuning_NeoBabel_mscoco_human = {}
        self.translations_cache_instruction_tuning_NeoBabel_dalle3 = {}
        self.translations_cache_instruction_tuning_NeoBabel_object_1 = {}
        self.translations_cache_instruction_tuning_NeoBabel_object_2 = {}
        self.translations_cache_instruction_tuning_NeoBabel_occupation_1 = {}
        self.translations_cache_instruction_tuning_NeoBabel_occupation_2 = {}
        self.translations_cache_instruction_tuning_NeoBabel_text_1 = {}
        self.translations_cache_instruction_tuning_NeoBabel_text_2 = {}
        self.lock = Lock()  # Create a lock to prevent multiple workers from loading the same file
        self.add_caption_prompt = add_caption_prompt
        if self.add_caption_prompt:
            with open("./training/questions.json") as f:
                self.caption_prompt = json.load(f)
                self.caption_prompt = ['USER: \n' + prompt + ' ASSISTANT:' for prompt in self.caption_prompt]
        else:
            self.caption_prompt = None

        if external_journeydb_caption_path != '':
            self.preload_translations_journeydb()
        else:
            self.translations_cache_journeydb = None
            print("No translations for journeydb")

        if external_laion12m_caption_path != '':
            self.preload_translations()
        else:
            self.translations_cache = {}
            print("No translations for laion12m")

        if external_caption_path != '':
            self.filenames2captions_sam1b = self._load_csv_file(self.external_caption_path)
        else:
            self.filenames2captions_sam1b = {}
            print("No captions for sam1b")

        if external_cc12m_caption_path != '':
            self.filenames2captions_cc12m = self._load_csv_file(self.external_cc12m_caption_path)
        else:
            self.filenames2captions_cc12m = {}
            print("No captions for cc12m")

        if external_crs3600_caption_path != '':
            self.preload_translations_crs3600(self.external_crs3600_caption_path)
        else:
            self.translations_cache_crs3600 = None
            print("No captions for crs3600")

        if external_liu4k_gcp_caption_path != '':
            self.preload_translations_liu4k_gcp(self.external_liu4k_gcp_caption_path)
        else:
            self.translations_cache_liu4k_gcp = {}
            print("No captions for liu4k_gcp")

        if external_open_img_pref_gcp_caption_path != '':
            self.preload_translations_open_img_pref_gcp(self.external_open_img_pref_gcp_caption_path)
        else:
            self.translations_cache_open_img_pref_gcp = {}
            print("No captions for open_img_pref_gcp")

        if external_open_pickapic_gcp_caption_path != '':
            # self.preload_translations_open_pickapic_gcp(self.external_open_pickapic_gcp_caption_path)
            self.preload_translations_pickapic()
        else:
            self.translations_cache_pickapic = None
            print("No captions for pickapic")

        if external_instruction_tuning_NeoBabel_geneval_train_caption_path != '':
            self.preload_translations_instruction_tuning_NeoBabel_geneval_train(self.external_instruction_tuning_NeoBabel_geneval_train_caption_path)
        else:
            self.translations_cache_instruction_tuning_NeoBabel_geneval_train = None
            print("No captions for instruction_tuning_NeoBabel_geneval_train")
            
        if external_instruction_tuning_NeoBabel_human_gestures_caption_path != '':
            self.preload_translations_instruction_tuning_NeoBabel_human_gestures(self.external_instruction_tuning_NeoBabel_human_gestures_caption_path)
        else:
            self.translations_cache_instruction_tuning_NeoBabel_human_gestures = None
            print("No captions for instruction_tuning_NeoBabel_human_gestures")
            
        if external_instruction_tuning_NeoBabel_journey_caption_path != '':
            self.preload_translations_instruction_tuning_NeoBabel_journey(self.external_instruction_tuning_NeoBabel_journey_caption_path)
        else:
            self.translations_cache_instruction_tuning_NeoBabel_journey = None
            print("No captions for instruction_tuning_NeoBabel_journey")
            
        if external_instruction_tuning_NeoBabel_mscoco_human_caption_path != '':
            self.preload_translations_instruction_tuning_NeoBabel_mscoco_human(self.external_instruction_tuning_NeoBabel_mscoco_human_caption_path)
        else:
            self.translations_cache_instruction_tuning_NeoBabel_mscoco_human = None
            print("No captions for instruction_tuning_NeoBabel_mscoco_human")
            
        if external_instruction_tuning_NeoBabel_dalle3_caption_path != '':
            self.preload_translations_instruction_tuning_NeoBabel_dalle3(self.external_instruction_tuning_NeoBabel_dalle3_caption_path)
        else:
            self.translations_cache_instruction_tuning_NeoBabel_dalle3 = None
            print("No captions for instruction_tuning_NeoBabel_dalle3")
            
        if external_instruction_tuning_NeoBabel_object_1_caption_path != '':
            self.preload_translations_instruction_tuning_NeoBabel_object_1(self.external_instruction_tuning_NeoBabel_object_1_caption_path)
        else:
            self.translations_cache_instruction_tuning_NeoBabel_object_1 = None
            print("No captions for instruction_tuning_NeoBabel_object_1")
            
        if external_instruction_tuning_NeoBabel_object_2_caption_path != '':
            self.preload_translations_instruction_tuning_NeoBabel_object_2(self.external_instruction_tuning_NeoBabel_object_2_caption_path)
        else:
            self.translations_cache_instruction_tuning_NeoBabel_object_2 = None
            print("No captions for instruction_tuning_NeoBabel_object_2")
            
        if external_instruction_tuning_NeoBabel_occupation_1_caption_path != '':
            self.preload_translations_instruction_tuning_NeoBabel_occupation_1(self.external_instruction_tuning_NeoBabel_occupation_1_caption_path)
        else:
            self.translations_cache_instruction_tuning_NeoBabel_occupation_1 = None
            print("No captions for instruction_tuning_NeoBabel_occupation_1")
        
        if external_instruction_tuning_NeoBabel_occupation_2_caption_path != '':
            self.preload_translations_instruction_tuning_NeoBabel_occupation_2(self.external_instruction_tuning_NeoBabel_occupation_2_caption_path)
        else:
            self.translations_cache_instruction_tuning_NeoBabel_occupation_2 = None
            print("No captions for instruction_tuning_NeoBabel_occupation_2")
            
        if external_instruction_tuning_NeoBabel_text_1_caption_path != '':
            self.preload_translations_instruction_tuning_NeoBabel_text_1(self.external_instruction_tuning_NeoBabel_text_1_caption_path)
        else:
            self.translations_cache_instruction_tuning_NeoBabel_text_1 = None
            print("No captions for instruction_tuning_NeoBabel_text_1")
            
        if external_instruction_tuning_NeoBabel_text_2_caption_path != '':
            self.preload_translations_instruction_tuning_NeoBabel_text_2(self.external_instruction_tuning_NeoBabel_text_2_caption_path)
        else:
            self.translations_cache_instruction_tuning_NeoBabel_text_2 = None
            print("No captions for instruction_tuning_NeoBabel_text_2")
        
        # Handle shard ratios
        if shard_ratios is None:
            shard_ratios = {"laion": 0.4, "jdb": 0.4, "internal": 0.2}
        assert abs(sum(shard_ratios.values()) - 1.0) < 1e-5, "Shard ratios must sum to 1.0"

        def tokenize(text):
            if tokenizer is not None:
                text = replace_person_token(text)
                input_ids = tokenizer(
                    text, max_length=max_seq_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                return input_ids[0]
            else:
                return text 

        # Expand shard lists. Accepts either a mapping with 'laion'/'jdb'/'internal'
        # groups (DictConfig or plain dict) or a flat list/str of shard urls,
        # which is treated as a single 'laion'-style group.
        if isinstance(train_shards_path_or_url, (omegaconf.dictconfig.DictConfig, dict)):
            laion_urls = train_shards_path_or_url.get("laion", [])
            jdb_urls = train_shards_path_or_url.get("jdb", [])
            internal_urls = train_shards_path_or_url.get("internal", [])
        else:
            if isinstance(train_shards_path_or_url, str):
                train_shards_path_or_url = [train_shards_path_or_url]
            laion_urls = list(train_shards_path_or_url)
            jdb_urls = []
            internal_urls = []

        laion_shards = list(itertools.chain.from_iterable(braceexpand(u) for u in laion_urls))
        jdb_shards = list(itertools.chain.from_iterable(braceexpand(u) for u in jdb_urls))
        internal_shards = internal_urls[:]


        
        processing_pipeline = [
            wds.decode("pil", handler=wds.ignore_and_continue),
            wds.map(self.load_external_caption, handler=wds.ignore_and_continue),
            wds.rename(
                images="jpg;png;jpeg;webp",
                input_ids="text;txt;caption",
                handler=wds.warn_and_continue,
            ),
            wds.map(filter_keys(set(["images", "input_ids"]))),
            wds.map(partial(image_transform, resolution=resolution), handler=wds.warn_and_continue),
            wds.map_dict(
                input_ids=tokenize,
                handler=wds.warn_and_continue,
            ),
        ]
    

        # Helper to build a per-group pipeline
        def build_group_pipeline(shards, buffer_size=1000):
            
            
            return wds.DataPipeline(
                wds.ResampledShards(shards),
                tarfile_to_samples_nothrow,
                wds.shuffle(buffer_size),
                *processing_pipeline,
            )


        # Create individual group datasets (only for non-empty groups:
        # ResampledShards asserts on empty shard lists)
        groups = [("laion", laion_shards), ("jdb", jdb_shards), ("internal", internal_shards)]
        groups = [(name, shards) for name, shards in groups if len(shards) > 0]
        assert groups, "Text2ImageDataset: no training shards provided"

        group_datasets = [build_group_pipeline(shards, buffer_size=shuffle_buffer_size)
                          for _, shards in groups]

        # Mix them according to shard_ratios (looked up by group name so config
        # key order doesn't matter); fall back to weighting by shard count.
        if shard_ratios is not None:
            probs = [shard_ratios[name] for name, _ in groups]
        else:
            probs = [len(shards) for _, shards in groups]
        mixed = wds.RandomMix(group_datasets, probs=probs)

        # Wrap mixed in DataPipeline to batch correctly
        final_ds = wds.DataPipeline(
            mixed,
            wds.batched(
                per_gpu_batch_size,
                partial=False,
                collation_fn=default_collate,
            ),
        )

        # Calculate epoch lengths and total samples
        num_batches = math.ceil(num_train_examples / global_batch_size)
        num_worker_batches = math.ceil(num_train_examples / (global_batch_size * num_workers))
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size

        # Wrap with epoch and loader
        self._train_dataset = final_ds.with_epoch(num_worker_batches)
        self._train_dataloader = wds.WebLoader(
            self._train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        self._train_dataloader.num_batches = num_batches
        self._train_dataloader.num_samples = num_samples

    def _load_csv_file(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Safely load CSV file with error handling."""
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # skip the header
            return {row[0]: row[1] for row in reader}

    def _process_caption(self, caption: str, is_captioning: bool = False) -> str:
        if not caption:
            return ''
            
        if is_captioning:
            if self.add_caption_prompt:
                prompt = random.choice(self.caption_prompt)
                return f"{prompt} {caption}"
            return caption
        
        # For generation
        caption = caption.split('.')[0] if random.random() < 0.5 else caption
        return remove_prefix(caption)

    def _get_language(self) -> str:
        if self.language == 'random':
            return random.choice(SUPPORTED_LANGUAGES)
        return self.language

    def _load_json_file(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Safely load JSON file with error handling."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading {filepath}: {e}")
            return {}

    def preload_translations(self):
        """Ensure only one worker loads each JSON file while others wait."""
        if not self.external_laion12m_caption_path:
            return

        # files = sorted(os.listdir(self.external_laion12m_caption_path))[:100]
        files = sorted(os.listdir(self.external_laion12m_caption_path))
        for shard_dir in tqdm(files, desc="Loading translations laion12m"):
            translations_file = os.path.join(self.external_laion12m_caption_path, shard_dir, "translations.json")
            
            if shard_dir not in self.translations_cache:  # Check before acquiring lock
                with self.lock:  # Lock ensures only one worker enters this block
                    if shard_dir not in self.translations_cache:  # Double-check after acquiring lock
                        if os.path.exists(translations_file):
                            with open(translations_file, "r") as f:
                                data = json.load(f)
                            # Create shared memory cache with unique name for this shard
                            shm_name = f"laion_{shard_dir}"
                            cache = SharedJSONCache(shm_name, data)
                            self.translations_cache[shard_dir] = shm_name
                        else:
                            # import pdb; pdb.set_trace()()
                            tsv_file = os.path.join(self.external_laion12m_caption_path, shard_dir, "results_internVL.tsv")
                            if os.path.exists(tsv_file):
                                print(f"Loading translations for laion {shard_dir} from TSV file")
                                with open(tsv_file, "r") as f:
                                    data = f.readlines()
                                # Create shared memory cache with unique name for this shard
                                # Convert TSV to dictionary format
                                data_dict = {}
                                for line in data[1:]:  # Skip header row
                                    try:
                                        # Skip empty lines
                                        if not line.strip():
                                            continue
                                        
                                        # Split line and handle different cases
                                        parts = line.strip().split('\t')
                                        if len(parts) == 0:  # Empty line after splitting
                                            # import pdb; pdb.set_trace()()
                                            continue
                                        elif len(parts) == 1:  # Only one part
                                            # import pdb; pdb.set_trace()()
                                            image_name = parts[0]
                                            caption = ""
                                        elif len(parts) > 2:  # More than two parts
                                            # import pdb; pdb.set_trace()()
                                            image_name = parts[0]
                                            caption = ""
                                            print(f"Warning: Line has {len(parts)} parts, using first part as image_name: {line.strip()}")
                                        else:  # Exactly two parts
                                            image_name, caption = parts
                                        
                                        # ... rest of your processing ...
                                        data_dict[image_name] = caption
                                    except Exception as e:
                                        print(f"Error processing line: {line.strip()}")
                                        print(f"Error details: {str(e)}")
                                        continue
                                data = {"English": data_dict}  # Replace TSV data with dictionary
                                shm_name = f"laion_{shard_dir}"
                                cache = SharedJSONCache(shm_name, data)
                                self.translations_cache[shard_dir] = shm_name
                            else:
                                print(f"No translations for laion {shard_dir}")
    def preload_translations_journeydb(self):
        """Ensure only one worker loads JourneyDB translations while others wait."""
        if not self.external_journeydb_caption_path:
            return

        # files = sorted(os.listdir(self.external_journeydb_caption_path))[:10]
        files = sorted(os.listdir(self.external_journeydb_caption_path))
        for shard in tqdm(files, desc="Loading translations journeydb"):
            shard_dir = shard.split('.')[0]
            translations_file = os.path.join(self.external_journeydb_caption_path, shard, "translations.json")

            if shard_dir not in self.translations_cache_journeydb:  # Check before locking
                with self.lock:  # Ensure only one worker enters at a time
                    if shard_dir not in self.translations_cache_journeydb:  # Double-check after locking
                        if os.path.exists(translations_file):
                            with open(translations_file, "r") as f:
                                data = json.load(f)
                            # Create shared memory cache with unique name for this shard
                            shm_name = f"journey_{shard_dir}"
                            cache = SharedJSONCache(shm_name, data)
                            self.translations_cache_journeydb[shard_dir] = shm_name

    def preload_translations_pickapic(self):
        if not self.external_open_pickapic_gcp_caption_path:
            return

        # load the json file and convert it to a dictionary
        with open(self.external_open_pickapic_gcp_caption_path, "r") as f:
            data = json.load(f)
        # Create shared memory cache with unique name for this shard
        shm_name = f"pickapic"
        cache = SharedJSONCache(shm_name, data)
        self.translations_cache_pickapic = shm_name
        print(f"pickapic data's keys: {data.keys()}")
        print(f"pickapic english data's keys: {list(data['English'].keys())[:10]}")
        print("pickapic data loaded")

    def preload_translations_instruction_tuning_NeoBabel_geneval_train(self, filepath):
        if not filepath:
            return
        
        with self.lock:
            if not self.translations_cache_instruction_tuning_NeoBabel_geneval_train:
                # read the json file and convert it to a dictionary
                with open(filepath, "r") as f:
                    data = json.load(f)
                # Create shared memory cache with unique name for this shard
                shm_name = f"Geneval_train"
                cache = SharedJSONCache(shm_name, data)
                self.translations_cache_instruction_tuning_NeoBabel_geneval_train = shm_name
                print(f"Geneval_train data's keys: {data.keys()}")
                print(f"instruction_tuning_NeoBabel_geneval_train english data's keys: {list(data['English'].keys())[:10]}")
                print("instruction_tuning_NeoBabel_geneval_train data loaded")

    def preload_translations_instruction_tuning_NeoBabel_human_gestures(self, filepath):
        if not filepath:
            return
        
        with self.lock:
            if not self.translations_cache_instruction_tuning_NeoBabel_human_gestures:
                # read the json file and convert it to a dictionary
                with open(filepath, "r") as f:
                    data = json.load(f)
                # Create shared memory cache with unique name for this shard
                shm_name = f"Human_gestures"
                cache = SharedJSONCache(shm_name, data)
                self.translations_cache_instruction_tuning_NeoBabel_human_gestures = shm_name
                print(f"Human_gestures data's keys: {data.keys()}")
                print(f"Human_gestures english data's keys: {list(data['English'].keys())[:10]}")
                print("Human_gestures data loaded")

    def preload_translations_instruction_tuning_NeoBabel_journey(self, filepath):
        if not filepath:
            return
        
        with self.lock:
            if not self.translations_cache_instruction_tuning_NeoBabel_journey:
                # read the json file and convert it to a dictionary
                with open(filepath, "r") as f:
                    data = json.load(f)
                # Create shared memory cache with unique name for this shard
                shm_name = f"Journey"
                cache = SharedJSONCache(shm_name, data)
                self.translations_cache_instruction_tuning_NeoBabel_journey = shm_name
                print(f"Journey data's keys: {data.keys()}")
                print(f"Journey english data's keys: {list(data['English'].keys())[:10]}")
                print("Journey data loaded")

    def preload_translations_instruction_tuning_NeoBabel_mscoco_human(self, filepath):
        if not filepath:
            return
        
        with self.lock:
            if not self.translations_cache_instruction_tuning_NeoBabel_mscoco_human:
                # read the json file and convert it to a dictionary
                with open(filepath, "r") as f:
                    data = json.load(f)
                # Create shared memory cache with unique name for this shard
                shm_name = f"MSCOCO_human"
                cache = SharedJSONCache(shm_name, data)
                self.translations_cache_instruction_tuning_NeoBabel_mscoco_human = shm_name
                print(f"MSCOCO_human data's keys: {data.keys()}")
                print(f"MSCOCO_human english data's keys: {list(data['English'].keys())[:10]}")
                print("MSCOCO_human data loaded")

    def preload_translations_instruction_tuning_NeoBabel_dalle3(self, filepath):
        if not filepath:
            return
        
        with self.lock:
            if not self.translations_cache_instruction_tuning_NeoBabel_dalle3:
                # read the json file and convert it to a dictionary
                with open(filepath, "r") as f:
                    data = json.load(f)
                # Create shared memory cache with unique name for this shard
                shm_name = f"dalle3"
                cache = SharedJSONCache(shm_name, data)
                self.translations_cache_instruction_tuning_NeoBabel_dalle3 = shm_name
                print(f"dalle3 data's keys: {data.keys()}")
                print(f"dalle3 english data's keys: {list(data['English'].keys())[:10]}")
                print("dalle3 data loaded")

    def preload_translations_instruction_tuning_NeoBabel_object_1(self, filepath):
        if not filepath:
            return
        
        with self.lock:
            if not self.translations_cache_instruction_tuning_NeoBabel_object_1:
                # read the json file and convert it to a dictionary
                with open(filepath, "r") as f:
                    data = json.load(f)
                # Create shared memory cache with unique name for this shard
                shm_name = f"object_1"
                cache = SharedJSONCache(shm_name, data)
                self.translations_cache_instruction_tuning_NeoBabel_object_1 = shm_name
                print(f"object_1 data's keys: {data.keys()}")
                print(f"object_1 english data's keys: {list(data['English'].keys())[:10]}")
                print("object_1 data loaded")
    
    def preload_translations_instruction_tuning_NeoBabel_object_2(self, filepath):
        if not filepath:
            return
        
        with self.lock:
            if not self.translations_cache_instruction_tuning_NeoBabel_object_2:
                # read the json file and convert it to a dictionary
                with open(filepath, "r") as f:
                    data = json.load(f)
                # Create shared memory cache with unique name for this shard
                shm_name = f"object_2"
                cache = SharedJSONCache(shm_name, data)
                self.translations_cache_instruction_tuning_NeoBabel_object_2 = shm_name
                print(f"object_2 data's keys: {data.keys()}")
                print(f"object_2 english data's keys: {list(data['English'].keys())[:10]}")
                print("object_2 data loaded")

    def preload_translations_instruction_tuning_NeoBabel_occupation_1(self, filepath):
        if not filepath:
            return
        
        with self.lock:
            if not self.translations_cache_instruction_tuning_NeoBabel_occupation_1:
                # read the json file and convert it to a dictionary
                with open(filepath, "r") as f:
                    data = json.load(f)
                # Create shared memory cache with unique name for this shard
                shm_name = f"occupation_1"
                cache = SharedJSONCache(shm_name, data)
                self.translations_cache_instruction_tuning_NeoBabel_occupation_1 = shm_name
                print(f"occupation_1 data's keys: {data.keys()}")
                print(f"occupation_1 english data's keys: {list(data['English'].keys())[:10]}")
                print("occupation_1 data loaded")
    
    def preload_translations_instruction_tuning_NeoBabel_occupation_2(self, filepath):
        if not filepath:
            return
        
        with self.lock:
            if not self.translations_cache_instruction_tuning_NeoBabel_occupation_2:
                # read the json file and convert it to a dictionary
                with open(filepath, "r") as f:
                    data = json.load(f)
                # Create shared memory cache with unique name for this shard
                shm_name = f"occupation_2"
                cache = SharedJSONCache(shm_name, data)
                self.translations_cache_instruction_tuning_NeoBabel_occupation_2 = shm_name
                print(f"occupation_2 data's keys: {data.keys()}")
                print(f"occupation_2 english data's keys: {list(data['English'].keys())[:10]}")
                print("occupation_2 data loaded")

    def preload_translations_instruction_tuning_NeoBabel_text_1(self, filepath):
        if not filepath:
            return
        
        with self.lock:
            if not self.translations_cache_instruction_tuning_NeoBabel_text_1:
                # read the json file and convert it to a dictionary
                with open(filepath, "r") as f:
                    data = json.load(f)
                # Create shared memory cache with unique name for this shard
                shm_name = f"text_1"
                cache = SharedJSONCache(shm_name, data)
                self.translations_cache_instruction_tuning_NeoBabel_text_1 = shm_name
                print(f"text_1 data's keys: {data.keys()}")
                print(f"text_1 english data's keys: {list(data['English'].keys())[:10]}")
                print("text_1 data loaded")

    def preload_translations_instruction_tuning_NeoBabel_text_2(self, filepath):
        if not filepath:
            return
        
        with self.lock:
            if not self.translations_cache_instruction_tuning_NeoBabel_text_2:
                # read the json file and convert it to a dictionary
                with open(filepath, "r") as f:
                    data = json.load(f)
                # Create shared memory cache with unique name for this shard
                shm_name = f"text_2"
                cache = SharedJSONCache(shm_name, data)
                self.translations_cache_instruction_tuning_NeoBabel_text_2 = shm_name
                print(f"text_2 data's keys: {data.keys()}")
                print(f"text_2 english data's keys: {list(data['English'].keys())[:10]}")
                print("text_2 data loaded")

    def preload_translations_crs3600(self, filepath):
        """Load CRS3600 captions into shared memory."""
        if not filepath:
            return
        
        # Use lock to ensure only one worker loads the data
        with self.lock:
            # self.translations_cache_crs3600 is an empty dictionary change the if
            if not self.translations_cache_crs3600:
                # read the json file and convert it to a dictionary
                with open(filepath, "r") as f:
                    data = json.load(f)
                # Create shared memory cache with unique name for this shard
                shm_name = f"crs3600"
                cache = SharedJSONCache(shm_name, data)
                self.translations_cache_crs3600 = shm_name

    def preload_translations_liu4k_gcp(self, filepath):
        """Load Liu4k GCP captions into shared memory."""
        if not filepath:
            return
        
        # Use lock to ensure only one worker loads the data
        with self.lock:
            if not self.translations_cache_liu4k_gcp:  # load once
                # read the tsv file and convert it to a dictionary
                with open(filepath, "r") as f:
                    data = f.readlines()

                # CONVERT THE TSV FILE TO A DICTIONARY
                data_dict = {}
                for line in data[1:]:  # Skip header row
                    parts = line.strip().split('\t')
                    if len(parts) == 0:
                        continue
                    elif len(parts) == 1:
                        image_name = parts[0]
                        caption = ""
                    elif len(parts) > 2:
                        image_name = parts[0]
                        caption = ""
                        print(f"Warning: Line has {len(parts)} parts, using first part as image_name: {line.strip()}")
                    else:  # Exactly two parts
                        image_name, caption = parts[0], parts[1]
                    data_dict[image_name] = caption
                # Create shared memory cache with unique name for this shard
                shm_name = f"liu4k_gcp"
                cache = SharedJSONCache(shm_name, data_dict)
                self.translations_cache_liu4k_gcp = shm_name

    def preload_translations_open_img_pref_gcp(self, filepath):
        """Load Open Image Pref GCP captions into shared memory."""
        if not filepath:
            return
        
        # Use lock to ensure only one worker loads the data
        with self.lock:
            if not self.translations_cache_open_img_pref_gcp:  # load once
                data_dict = {}
                # load the csv file and convert it to a dictionary
                with open(filepath, "r") as f:
                    reader = csv.reader(f)
                    next(reader, None)  # Skip header if present
                    for row in reader:
                        if len(row) == 0:
                            continue
                        elif len(row) == 1:
                            image_name = row[0]
                            caption = ""
                        elif len(row) >= 2:
                            image_name = row[0]
                            caption = row[1]
                        data_dict[image_name] = caption
                
                # Create shared memory cache with unique name
                shm_name = "open_img_pref_gcp"
                cache = SharedJSONCache(shm_name, data_dict)
                self.translations_cache_open_img_pref_gcp = shm_name

    def preload_translations_open_pickapic_gcp(self, filepath):
        """Load Open Pickapic GCP captions into shared memory."""
        if not filepath:
            return
        
        with self.lock:
            if not self.translations_cache_open_pickapic_gcp:  # load once
                data_dict = {}
                # load the csv file and convert it to a dictionary
                with open(filepath, "r") as f:
                    reader = csv.reader(f)
                    next(reader, None)  # Skip header if present
                    for row in reader:
                        if len(row) == 0:
                            continue
                        elif len(row) == 1:
                            image_name = row[0]
                            caption = ""
                        elif len(row) >= 2:
                            image_name = row[0]
                            caption = row[1]
                        data_dict[image_name] = caption
                
                # Create shared memory cache with unique name
                shm_name = "open_pickapic_gcp"
                cache = SharedJSONCache(shm_name, data_dict)
                self.translations_cache_open_pickapic_gcp = shm_name

    def load_external_caption(self, sample):
        if 'txt' not in sample.keys():
            sample['txt'] = ''

        key = sample['__key__']
        url = sample.get('__url__', '')
        

        # Handle different data sources
        if 'SA1B' in url:
            filename = sample['__key__'].split('/')[-1]
            filename = filename + '.jpg'
            
            if filename in self.filenames2captions_sam1b:
                captions = self.filenames2captions_sam1b[filename]
            else:
                captions = ""
                # print(f"No captions for SA1B {filename}")
            sample['txt'] = self._process_caption(captions, self.is_captioning)
            return sample

        # ... existing code ...
        elif 'laion' in url:
            shard_name = url.split('/')[-1].split('.')[0]
            key = str(int(key))
            language = self._get_language()

            shm_name = self.translations_cache.get(shard_name)
            if shm_name:
                cache = SharedJSONCache.attach(shm_name)
                _data = cache.get_data()
                captions = _data.get(language, _data.get("English", {})).get(key, '')
                cache.shm.close()  # Important: close but don't unlink
            else:
                captions = ''
                print(f"No translations for laion {shard_name}")
            
            sample['txt'] = self._process_caption(captions, self.is_captioning)
            return sample

        elif 'pickapic' in sample['__url__']:
            filename = sample["__key__"]
            if self.translations_cache_pickapic:
                shm_name = "pickapic"
                language = self._get_language()
                cache = SharedJSONCache.attach(shm_name)
                _data = cache.get_data()
                captions = _data.get(language, _data.get("English", {})).get(filename, '')
                cache.shm.close()  # Important: close but don't unlink
            else:
                captions = ""
                print(f"No translations for pickapic")
            sample['txt'] = self._process_caption(captions, self.is_captioning)
            return sample

        elif "Geneval_train" in url:
            filename = sample["__key__"] + ".jpg"
            if self.translations_cache_instruction_tuning_NeoBabel_geneval_train:
                shm_name = "Geneval_train"
                cache = SharedJSONCache.attach(shm_name)
                language = self._get_language()
                _data = cache.get_data()
                captions = _data.get(language, _data.get("English", {})).get(filename, '')
                cache.shm.close()  # Important: close but don't unlink
            else:
                captions = ""
                print(f"No translations for Geneval_train")
            sample['txt'] = self._process_caption(captions, self.is_captioning)
            return sample

        elif "Human_gestures" in url:
            filename = sample["__key__"] + ".jpg"
            if self.translations_cache_instruction_tuning_NeoBabel_human_gestures:
                shm_name = "Human_gestures"
                cache = SharedJSONCache.attach(shm_name)
                language = self._get_language()
                _data = cache.get_data()
                captions = _data.get(language, _data.get("English", {})).get(filename, '')
                cache.shm.close()  # Important: close but don't unlink
            else:
                captions = ""
                print(f"No translations for Human_gestures")
            sample['txt'] = self._process_caption(captions, self.is_captioning)
            return sample
        
        elif "instruction_tuning_maya/Journey" in url:
            filename = sample["__key__"] + ".jpg"
            if self.translations_cache_instruction_tuning_NeoBabel_journey:
                shm_name = "Journey"
                cache = SharedJSONCache.attach(shm_name)
                language = self._get_language()
                _data = cache.get_data()
                captions = _data.get(language, _data.get("English", {})).get(filename, '')
                cache.shm.close()  # Important: close but don't unlink
            else:
                captions = ""
                print(f"No translations for Journey")
            sample['txt'] = self._process_caption(captions, self.is_captioning)
            return sample
        
        elif "MSCOCO_human" in url:
            filename = sample["__key__"] + ".jpg"
            if self.translations_cache_instruction_tuning_NeoBabel_mscoco_human:
                shm_name = "MSCOCO_human"
                cache = SharedJSONCache.attach(shm_name)
                language = self._get_language()
                _data = cache.get_data()
                captions = _data.get(language, _data.get("English", {})).get(filename, '')
                cache.shm.close()  # Important: close but don't unlink
            else:
                captions = ""
                print(f"No translations for MSCOCO_human")
            sample['txt'] = self._process_caption(captions, self.is_captioning)
            return sample
        
        elif "dalle3" in url:
            filename = sample["__key__"] + ".jpg"
            if self.translations_cache_instruction_tuning_NeoBabel_dalle3:
                shm_name = "dalle3"
                cache = SharedJSONCache.attach(shm_name)
                language = self._get_language()
                _data = cache.get_data()
                captions = _data.get(language, _data.get("English", {})).get(filename, '')
                cache.shm.close()  # Important: close but don't unlink
            else:
                captions = ""
                print(f"No translations for dalle3")
            sample['txt'] = self._process_caption(captions, self.is_captioning)
            return sample

        elif "object_1" in url:
            filename = sample["__key__"] + ".jpg"
            if self.translations_cache_instruction_tuning_NeoBabel_object_1:
                shm_name = "object_1"
                cache = SharedJSONCache.attach(shm_name)
                language = self._get_language()
                _data = cache.get_data()
                captions = _data.get(language, _data.get("English", {})).get(filename, '')
                cache.shm.close()  # Important: close but don't unlink
            else:
                captions = ""
                print(f"No translations for object_1")
            sample['txt'] = self._process_caption(captions, self.is_captioning)
            return sample

        elif "object_2" in url:
            filename = sample["__key__"] + ".jpg"
            if self.translations_cache_instruction_tuning_NeoBabel_object_2:
                shm_name = "object_2"
                cache = SharedJSONCache.attach(shm_name)
                language = self._get_language()
                _data = cache.get_data()
                captions = _data.get(language, _data.get("English", {})).get(filename, '')
                cache.shm.close()  # Important: close but don't unlink
            else:
                captions = ""
                print(f"No translations for object_2")
            sample['txt'] = self._process_caption(captions, self.is_captioning)
            return sample
        
        elif "occupation_1" in url:
            filename = sample["__key__"] + ".jpg"
            if self.translations_cache_instruction_tuning_NeoBabel_occupation_1:
                shm_name = "occupation_1"
                cache = SharedJSONCache.attach(shm_name)
                language = self._get_language()
                _data = cache.get_data()
                captions = _data.get(language, _data.get("English", {})).get(filename, '')
                cache.shm.close()  # Important: close but don't unlink
            else:
                captions = ""
                print(f"No translations for occupation_1")
            sample['txt'] = self._process_caption(captions, self.is_captioning)
            return sample
        
        elif "occupation_2" in url:
            filename = sample["__key__"] + ".jpg"
            if self.translations_cache_instruction_tuning_NeoBabel_occupation_2:
                shm_name = "occupation_2"
                cache = SharedJSONCache.attach(shm_name)
                language = self._get_language()
                _data = cache.get_data()
                captions = _data.get(language, _data.get("English", {})).get(filename, '')
                cache.shm.close()  # Important: close but don't unlink
            else:
                captions = ""
                print(f"No translations for occupation_2")
            sample['txt'] = self._process_caption(captions, self.is_captioning)
            return sample
        
        elif "text_1" in url:
            filename = sample["__key__"] + ".jpg"
            if self.translations_cache_instruction_tuning_NeoBabel_text_1:
                shm_name = "text_1"
                cache = SharedJSONCache.attach(shm_name)
                language = self._get_language()
                _data = cache.get_data()
                captions = _data.get(language, _data.get("English", {})).get(filename, '')
                cache.shm.close()  # Important: close but don't unlink
            else:
                captions = ""
                print(f"No translations for text_1")
            sample['txt'] = self._process_caption(captions, self.is_captioning)
            return sample

        elif "text_2" in url:
            filename = sample["__key__"] + ".jpg"
            if self.translations_cache_instruction_tuning_NeoBabel_text_2:
                shm_name = "text_2"
                cache = SharedJSONCache.attach(shm_name)
                language = self._get_language()
                _data = cache.get_data()
                captions = _data.get(language, _data.get("English", {})).get(filename, '')
                cache.shm.close()  # Important: close but don't unlink
            else:
                captions = ""
                print(f"No translations for text_2")
            sample['txt'] = self._process_caption(captions, self.is_captioning)
            return sample

        elif 'cc12m' in sample['__url__']:
            filename = sample["__key__"] + ".jpg"
            if filename in self.filenames2captions_cc12m:
                captions = self.filenames2captions_cc12m[filename]
            else:
                captions = ""
                # print(f"No captions for CC12M {filename}")
            sample['txt'] = self._process_caption(captions, self.is_captioning)
            return sample

        # In load_external_caption method:
        elif "DB" in sample['__url__']:
            shard_name = sample['__key__'].split('/')[0]
            key = sample['__key__']
            language = self._get_language()

            shm_name = self.translations_cache_journeydb.get(shard_name)
            if shm_name:
                cache = SharedJSONCache.attach(shm_name)
                captions = cache.get_data().get(language, {}).get(key, '')
                # print(f"captions: {captions}")
                cache.shm.close()  # Important: close but don't unlink
            else:
                captions = ''
                print(f"No translations for journeydb {shard_name}, key: {key}, url: {url}")
            sample['txt'] = captions
            return sample

        elif 'crs' in sample['__url__']:
            filename = sample["__key__"].split('/')[-1].split('.')[0]
            language = self._get_language()
            # Use shared memory instead of filenames2captions_crs3600
            if self.translations_cache_crs3600:
                cache = SharedJSONCache.attach(self.translations_cache_crs3600)
                captions = cache.get_data().get(language, {}).get(filename, '')
                cache.shm.close()  # Important: close but don't unlink
            else:
                captions = ""
                print(f"No translations for crs3600")
            sample['txt'] = self._process_caption(captions, self.is_captioning)
            return sample

        elif 'liu' in sample['__url__']:
            filename = sample["__key__"].split('/')[-1].split('.')[0]
            if self.translations_cache_liu4k_gcp:
                cache = SharedJSONCache.attach(self.translations_cache_liu4k_gcp)
                captions = cache.get_data().get(filename, '')
                cache.shm.close()  # Important: close but don't unlink
            else:
                captions = ""
                print(f"No translations for liu4k_gcp")
            sample['txt'] = self._process_caption(captions, self.is_captioning)
            return sample

        elif 'pref' in sample['__url__']:
            filename = sample["__key__"].split('/')[-1].split('.')[0] + ".png"
            if self.translations_cache_open_img_pref_gcp:
                cache = SharedJSONCache.attach(self.translations_cache_open_img_pref_gcp)
                captions = cache.get_data().get(filename, '')
                cache.shm.close()  # Important: close but don't unlink
            else:
                captions = ""
                print(f"No translations for open_img_pref_gcp")
            sample['txt'] = self._process_caption(captions, self.is_captioning)
            return sample
        # elif 'pickapic' in sample['__url__']:
        #     filename = sample["__key__"]
        #     if self.translations_cache_open_pickapic_gcp:
        #         cache = SharedJSONCache.attach(self.translations_cache_open_pickapic_gcp)
        #         captions = cache.get_data().get(filename, '')
        #         cache.shm.close()  # Important: close but don't unlink
        #     else:
        #         captions = ""
        #         print(f"No translations for open_pickapic_gcp")
        #     sample['txt'] = self._process_caption(captions, self.is_captioning)
        #     return sample
        else:
            return sample
    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader


    def __del__(self):
        # Cleanup shared memory (caches may be None or plain strings depending on config)
        for cache in (getattr(self, "translations_cache", None),
                      getattr(self, "translations_cache_journeydb", None)):
            if not isinstance(cache, dict):
                continue
            for shm_name in cache.values():
                try:
                    SharedMemory(name=shm_name).unlink()
                except FileNotFoundError:
                    pass


if __name__ == '__main__':
    pass