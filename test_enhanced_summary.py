import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'local-voice-to-voice', 'src'))
from pipeline.conversation_summarizer import ConversationSummarizer
from llm.gemini_llm import GeminiLLM

# Initialize
llm = GeminiLLM()
summarizer = ConversationSummarizer(llm)

# Test conversation with personal facts
test_conversation = [
    {'role': 'user', 'content': 'Hi, my name is Alex and I love the color teal. I work as a software engineer in Seattle.'},
    {'role': 'assistant', 'content': 'Nice to meet you Alex! Tell me more about your work in Seattle.'},
    {'role': 'user', 'content': 'I have two cats and I am 28 years old. My birthday is March 15th.'},
    {'role': 'assistant', 'content': 'Cats are wonderful! What are their names?'},
    {'role': 'user', 'content': 'Whiskers and Mittens. I also struggle with depression sometimes.'}
]

print('Testing enhanced conversation summarization...')
summary = summarizer.summarize_conversation(test_conversation)
print('\n' + '='*60)
print('ENHANCED SUMMARY RESULT:')
print('='*60)
if summary:
    print(summary[0]['content'])
else:
    print('No summary generated') 