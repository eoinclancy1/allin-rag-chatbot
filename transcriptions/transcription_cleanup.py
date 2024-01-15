# nicer code - but replaced by transcription_cleanup_V2.py
import pandas as pd
import re

def process_text_file_v3(file_path, output_csv_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Regular expression to match sentences ending with a question mark
    pattern = r'([^.!?]*\?)([^?]*)(?=\?)'

    # Find all matches
    matches = re.findall(pattern, content)

    # Preparing the data for the DataFrame
    data = []
    for match in matches:
        question = match[0].strip()
        answer = match[1].strip()

        # Splitting the answer into sentences
        answer_sentences = re.split(r'[.!?]', answer)
        answer_sentences = [sentence.strip() for sentence in answer_sentences if sentence.strip()]

        # Joining the sentences with line breaks
        formatted_answer = '\n'.join(answer_sentences)

        data.append([question, formatted_answer])

    # Creating the DataFrame
    df = pd.DataFrame(data, columns=['Question', 'Answer'])

    # Writing the DataFrame to a CSV file
    df.to_csv(output_csv_path, index=False)

    return f"CSV file created at {output_csv_path}"

# Example usage
process_text_file_v3('transcriptions/phillips.txt', 'phillips3.csv')