import os
import json
import pandas as pd
from google import genai
from google.genai import types
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = "INSERT_YOUR_API_KEY_HERE"

class GeminiAPIError(Exception):
    """Custom exception for Gemini API errors."""
    pass

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def process_caption_with_retry(client, caption):
    model = "gemini-2.0-flash-lite"
    prompt = f"""You are a machine translation. I write a text in English and try to translate it into Persian, French, Dutch, Hindi, and Chinese (simplified). If there is any text in quotation marks, do not translate that part and just use the original English text.\n\n{caption}"""
    
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]
    
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.OBJECT,
            properties={
                "Persian": genai.types.Schema(type=genai.types.Type.STRING),
                "French": genai.types.Schema(type=genai.types.Type.STRING),
                "Dutch": genai.types.Schema(type=genai.types.Type.STRING),
                "Hindi": genai.types.Schema(type=genai.types.Type.STRING),
                "Chinese (simplified)": genai.types.Schema(type=genai.types.Type.STRING),
            },
        ),
    )

    time.sleep(0.5)
    
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    
    if not response.text:
        print(f"Empty response received from Gemini API for caption: {caption}")
        return json.dumps({
            "Persian": "",
            "French": "",
            "Dutch": "",
            "Hindi": "",
            "Chinese (simplified)": "",
        })

    return response.text

def save_progress(result, processed_ids, checkpoint_file, progress_file):
    """Save both results and progress tracking"""
    # Save translations
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # Save progress tracking
    with open(progress_file, 'w') as f:
        json.dump(list(processed_ids), f)

def load_progress(checkpoint_file, progress_file):
    """Load previous progress"""
    result = {lang: {} for lang in [
        "English", "Western Persian", "Dutch", 
        "French", "Hindi", "Chinese (Simplified)"
    ]}
    processed_ids = set()
    
    if os.path.exists(checkpoint_file) and os.path.exists(progress_file):
        try:
            # Load translations
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
            # Load progress
            with open(progress_file, 'r') as f:
                processed_ids = set(json.load(f))
            print(f"Resumed from checkpoint with {len(processed_ids)} processed items")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    return result, processed_ids

def generate(input_csv: str, save_frequency: int = 10):
    # Initialize Gemini client
    client = genai.Client(
        api_key=API_KEY,
    )
    
    # Setup checkpoint files
    input_dir = os.path.dirname(input_csv)
    checkpoint_file = os.path.join(input_dir, 'translations_gemini_checkpoint.json')
    progress_file = os.path.join(input_dir, 'translation_progress.json')
    
    # Load previous progress
    result, processed_ids = load_progress(checkpoint_file, progress_file)

    # Read CSV file
    df = pd.read_csv(input_csv)
    total_items = len(df)
    
    # Store original English captions if not already loaded
    for idx, row in df.iterrows():
        image_name = f"{row['image_id']}"
        if image_name not in result["English"]:
            result["English"][image_name] = row['caption']
    
    print(f"Starting/Resuming translation of {total_items} items...")
    print(f"Already processed: {len(processed_ids)} items")
    
    error_count = 0
    consecutive_errors = 0
    
    # Define requests per second
    requests_per_second = 4000 // 60  # Approximately 67 requests per second

    # Initialize successful request counter and start time
    successful_requests = 0
    start_time = time.time()
    stop_processing = False

    # Use ThreadPoolExecutor to process captions concurrently
    with ThreadPoolExecutor(max_workers=requests_per_second) as executor:
        futures = []
        future_to_image_name = {}

        def drain_futures(pending_futures):
            """Harvest a batch of futures; returns True if processing should stop."""
            nonlocal error_count, consecutive_errors, successful_requests, start_time
            stop = False
            for future in as_completed(pending_futures):
                image_name = future_to_image_name[future]
                try:
                    translations = json.loads(future.result())

                    # Store translations
                    result["Western Persian"][image_name] = translations["Persian"]
                    result["Dutch"][image_name] = translations["Dutch"]
                    result["French"][image_name] = translations["French"]
                    result["Hindi"][image_name] = translations["Hindi"]
                    result["Chinese (Simplified)"][image_name] = translations["Chinese (simplified)"]

                    processed_ids.add(image_name)
                    consecutive_errors = 0  # Reset error counter on success
                    successful_requests += 1  # Increment successful request counter

                    # Save progress periodically
                    if len(processed_ids) % save_frequency == 0:
                        save_progress(result, processed_ids, checkpoint_file, progress_file)
                        print(f"\nProgress saved: {len(processed_ids)}/{total_items} items processed")

                except Exception as e:
                    print(f"\nError processing caption for image {image_name}: {e}")
                    error_count += 1
                    consecutive_errors += 1

                    # Save progress on error
                    save_progress(result, processed_ids, checkpoint_file, progress_file)
                    print(f"Progress saved after error. {len(processed_ids)}/{total_items} items processed")

                    if consecutive_errors >= 3:
                        print("Too many consecutive errors, taking a longer break...")
                        time.sleep(60)  # 1 minute break
                        consecutive_errors = 0

                    if error_count > 50:
                        print("Too many total errors, stopping processing")
                        stop = True

            # Log successful requests per minute
            elapsed_time = time.time() - start_time
            if elapsed_time >= 60:
                print(f"Successful requests in the last minute: {successful_requests}")
                successful_requests = 0
                start_time = time.time()

            return stop

        for idx, row in df.iterrows():
            image_name = f"{row['image_id']}"
            if image_name in processed_ids or pd.isna(row['caption']):
                print(f"Skipping {image_name} (already processed or caption is NaN)")
                continue

            future = executor.submit(process_caption_with_retry, client, row['caption'])
            futures.append(future)
            future_to_image_name[future] = image_name

            # Process in batches of requests_per_second
            if len(futures) >= requests_per_second:
                stop_processing = drain_futures(futures)
                futures.clear()
                if stop_processing:
                    break
                time.sleep(1)  # Wait for 1 second before the next batch

        # Drain any leftover futures from the final partial batch
        if futures and not stop_processing:
            drain_futures(futures)
            futures.clear()
        print("Finished processing all batches.")
    
    # Save final results
    translation_file = os.path.join(input_dir, 'translations_gemini.json')

    with open(translation_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # Clean up checkpoint files only if complete
    if len(processed_ids) == total_items:
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        if os.path.exists(progress_file):
            os.remove(progress_file)
        print("\nProcessing completed successfully!")
    else:
        print("\nProcessing stopped before completion.")
        print("Checkpoint files preserved for future resume.")
    
    print(f"Translations saved to {translation_file}")
    print(f"Total items processed: {len(processed_ids)}/{total_items}")
    print(f"Total errors encountered: {error_count}")

if __name__ == "__main__":
    import fire
    fire.Fire(generate)