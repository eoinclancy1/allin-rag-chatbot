# INPUT: raw transcript as a text file
# OUTPUT: CSV with columns for questions and answers
# PROCESS: Pretty crude, but it tries to split the raw text on '?', with the assumption that it'll be pretty good at separating the interview into Q&As without losing context or summarizing the text
# FOLLOW-UP: Once the csv files have been generated, manual review and cleanup is then required to create the "clean_x" versions stored in ./clean_transcriptions
#            To maintain accuracy in the transcript that'll get embedded, this was found to be the best workflow. 
#            Independent of the S2T model used, there were some errors in transcription, so a human in the loop was necessary.
#            Similarly, given the nature of the multi-way conversations used as input, questions certainly needed review as not all were posed to the political candidate
import csv
import re

def process_interview(input_file, output_file):
    # Read the content of the file
    with open(input_file, 'r') as file:
        text = file.read()

    # Splitting the text into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)

    # Initializing an empty list to store question-answer pairs
    qa_pairs = []
    answer = ""
    question = ""

    for sentence in sentences:
        if sentence.endswith('?'):
            if answer:
                # Append the previous Q&A pair to the list if there's a question already
                if question:
                    qa_pairs.append([question, answer.strip()])
                # Reset answer for the next pair
                answer = ""
            # Current sentence is the new question
            question = sentence
        else:
            # Add sentence to the answer
            answer += sentence + " "

    # Append the last Q&A pair if there's an unanswered question
    if question and answer:
        qa_pairs.append([question, answer.strip()])

    # Write the Q&A pairs to a CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Question', 'Answer'])  # Writing header
        writer.writerows(qa_pairs)

# Example usage
process_interview('transcriptions/christie.txt', 'christiev2_1.csv')

# These files have since been moved to ./automated_QA_csv
