# Multilingual Expansion of Benchmarks

`translate.py` is the script utilized to translate english **DPG** and **GenEval** into multilingual prompts. It translates captions from a CSV file into Persian, French, Dutch, Hindi, and Simplified Chinese using the Google Gemini API, with support for batching, retries, and checkpointing.

## 📥 Input

* A CSV file with:

  * `image_id`
  * `caption` (English text)
* Google Gemini API key

## 📤 Output

* `translations_gemini.json`: JSON with original and translated captions
* Checkpoint files:

  * `translations_gemini_checkpoint.json`
  * `translation_progress.json` (for resuming)

## Installation

Install the required packages:

```bash
pip install pandas google-genai tqdm tenacity fire
```

## Usage

```bash
python translate.py --input_csv path/to/file.csv
```

## Notes
- The script is designed for high-throughput translation and may use a large number of threads. Adjust `requests_per_second` in the script if needed.
- If interrupted, simply rerun the script with the same arguments to resume.


**In order to extend this to other multilingual prompts, please modify the prompt and `GenerateContentConfig` as required.**
