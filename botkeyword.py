import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# --- Step 1: Securely Load Your Google AI API Key ---
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise ValueError("Google AI API key not found. Please set it in your .env file as GOOGLE_API_KEY=your_key_here")

# Configure the library with your key
genai.configure(api_key=google_api_key)


def extract_structured_legal_info(text_content):
    """
    Uses the Google Gemini API to extract legal keywords, terminologies, and entities.
    Returns a Python dictionary parsed from the API's JSON response.
    """
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    prompt = f"""
    As a specialized legal assistant bot for 'LegalSathi', your task is to analyze the following legal document text.
    Your goal is to identify and extract all significant keywords, legal terminology, and legal entities.

    Please categorize your findings into three distinct lists:
    1.  **keywords**: General important topics or subjects (e.g., "breach of contract", "intellectual property", "negligence").
    2.  **legal_terminologies**: Specific legal terms, phrases, or Latin maxims (e.g., "Res Judicata", "Caveat Emptor", "writ of mandamus").
    3.  **legal_entities**: Named parties, courts, acts, sections, or specific roles (e.g., "Plaintiff", "Defendant", "Supreme Court of India", "Indian Penal Code, Section 302").

    Return the result as a single, valid JSON object with the keys "keywords", "legal_terminologies", and "legal_entities". Do not include any text before or after the JSON object.

    --- DOCUMENT TEXT ---
    {text_content}
    --- END OF DOCUMENT ---

    JSON Output:
    """
    try:
        response = model.generate_content(prompt)
        # Clean up the response to ensure it's valid JSON
        cleaned_response_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_response_text)
    
    except json.JSONDecodeError as e:
        print(f"❌ JSON Parsing Error: {e}")
        print(f"   Raw response from API was: {response.text}")
        return None
    except Exception as e:
        print(f"An error occurred with the Google Gemini API: {e}")
        return None


def main():
    """
    Main function to run the bot on a directory of files, saving JSON outputs to a separate folder.
    """
    # --- Configuration ---
    # Path to the folder containing your .txt files
    database_folder = r"D:\LegalSathi\legalSatthi\Database"
    
    # Path to the folder where you want to save the .json files
    # ❗️ IMPORTANT: Change this path to your desired output location.
    output_folder = r"D:\LegalSathi\legalSatthi\JsonOutput" 

    # --- End of Configuration ---

    if not os.path.isdir(database_folder):
        print(f"❌ ERROR: The source folder does not exist: {database_folder}")
        return

    # Create the output directory if it doesn't exist
    try:
        os.makedirs(output_folder, exist_ok=True)
        print(f"📁 JSON files will be saved in: {output_folder}")
    except OSError as e:
        print(f"❌ ERROR: Could not create output directory {output_folder}: {e}")
        return

    print(f"📂 Scanning for .txt files in: {database_folder}\n")
    
    files_processed = 0
    for filename in os.listdir(database_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(database_folder, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    document_text = file.read()
                
                if not document_text.strip():
                    print(f"📄 Skipping empty file: {filename}\n")
                    continue

                print(f"--- Processing File: {filename} ---")
                print("🤖 Contacting Google AI for structured data extraction...")

                extracted_data = extract_structured_legal_info(document_text)

                if extracted_data:
                    # Create the output filename by replacing .txt with .json
                    base_filename, _ = os.path.splitext(filename)
                    # Construct the full path to the output file in the specified output folder
                    output_json_path = os.path.join(output_folder, f"{base_filename}.json")

                    # Write the extracted dictionary to its own JSON file
                    with open(output_json_path, 'w', encoding='utf-8') as f:
                        json.dump(extracted_data, f, indent=4)
                    
                    print(f"✅ Successfully saved extracted data to: {output_json_path}")
                    files_processed += 1
                else:
                    print(f"⚠️ Failed to extract data for {filename}.")
                
                print(f"--- Finished with {filename} ---\n")

            except Exception as e:
                print(f"❌ An unexpected error occurred while processing {filename}: {e}\n")
    
    print(f"🎉 All done! Processed {files_processed} file(s).")


if __name__ == "__main__":
    main()
