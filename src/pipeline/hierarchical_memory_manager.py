from typing import List, Dict, Optional
import os
from pathlib import Path
from .conversation_logger import ConversationLogger
from .conversation_summarizer import ConversationSummarizer

# Constants for memory hierarchy
NUM_CONV_SUMMARIES_FOR_STM = 5
NUM_STM_FOR_LTM = 5

class HierarchicalMemoryManager:
    """
    Manages the creation of Short-Term Memory (STM) and Long-Term Memory (LTM)
    summaries from individual conversation summaries and STMs respectively.
    """
    def __init__(self, conversation_logger: ConversationLogger, conversation_summarizer: ConversationSummarizer):
        self.logger = conversation_logger
        self.summarizer = conversation_summarizer

    def update_memory_hierarchy(self):
        """
        Processes new conversation summaries to create STMs,
        and new STMs to create LTMs.
        """
        print("Updating memory hierarchy...")
        self._create_stm_summaries_if_needed()
        self._create_ltm_summaries_if_needed()
        print("Memory hierarchy update complete.")

    def _get_processed_files_from_meta_summaries(self, meta_summary_files: List[str], key_for_constituent_files: str) -> set:
        """
        Helper function to extract a set of all constituent files that have already been processed
        into a higher-level summary (STM or LTM).
        Example: For STMs, key_for_constituent_files is 'constituent_summaries'.
                  For LTMs, key_for_constituent_files is 'constituent_stms'.
        """
        processed_files = set()
        for meta_file_path in meta_summary_files:
            try:
                meta_summary_data = {}
                if key_for_constituent_files == 'constituent_summaries': # STM file
                    meta_summary_data = self.logger.load_stm_summary_file(meta_file_path)
                elif key_for_constituent_files == 'constituent_stms': # LTM file
                    meta_summary_data = self.logger.load_ltm_summary_file(meta_file_path)
                
                if meta_summary_data and key_for_constituent_files in meta_summary_data:
                    for f_path in meta_summary_data[key_for_constituent_files]:
                        processed_files.add(str(Path(f_path))) # Normalize path for comparison
            except Exception as e:
                print(f"Error loading or parsing meta summary file {meta_file_path}: {e}")
        return processed_files

    def _create_stm_summaries_if_needed(self):
        """
        Checks for new individual conversation summaries and processes them into STMs.
        """
        print("Checking for new conversation summaries to process into STMs...")
        all_conv_summary_files = [str(Path(f)) for f in self.logger.get_summary_files()]
        
        # Include old summaries as well, if any, as they might not have been processed into STMs
        old_summaries_dir = self.logger.log_dir / "old_summaries"
        if old_summaries_dir.exists():
            old_summary_files = [str(Path(f)) for f in old_summaries_dir.glob("conversation_*_summary.json")]
            all_conv_summary_files.extend(old_summary_files)
        
        # Remove duplicates that might arise from including old_summaries
        all_conv_summary_files = sorted(list(set(all_conv_summary_files)))


        stm_files = self.logger.get_stm_summary_files()
        processed_conv_summary_files = self._get_processed_files_from_meta_summaries(stm_files, 'constituent_summaries')

        new_conv_summary_files = [
            f for f in all_conv_summary_files if f not in processed_conv_summary_files
        ]
        
        # Sort by inferred timestamp from filename (newest first for processing, then reverse for batching)
        # This ensures we process older summaries first when batching.
        new_conv_summary_files.sort(key=lambda f: Path(f).stem, reverse=True)


        if not new_conv_summary_files:
            print("No new conversation summaries to process for STMs.")
            return

        print(f"Found {len(new_conv_summary_files)} new conversation summaries for STM creation.")

        # Reverse for batching so older ones are processed first
        new_conv_summary_files.reverse() 

        for i in range(0, len(new_conv_summary_files), NUM_CONV_SUMMARIES_FOR_STM):
            batch_to_process = new_conv_summary_files[i:i + NUM_CONV_SUMMARIES_FOR_STM]
            if len(batch_to_process) < NUM_CONV_SUMMARIES_FOR_STM:
                print(f"Skipping STM creation for batch of {len(batch_to_process)} summaries (less than {NUM_CONV_SUMMARIES_FOR_STM}). Will process later.")
                continue

            print(f"Processing batch of {len(batch_to_process)} conversation summaries for STM...")
            constituent_files_for_stm = [str(Path(f)) for f in batch_to_process] # Ensure full paths
            
            combined_content = ""
            for summary_file_path_str in batch_to_process:
                try:
                    summary_data = self.logger.load_summary_file(summary_file_path_str)
                    # Extract content from the 'messages' list, assuming it's an assistant's message
                    if summary_data.get('messages') and isinstance(summary_data['messages'], list):
                        for msg in summary_data['messages']:
                            if msg.get('role') == 'assistant' and msg.get('content'):
                                combined_content += f"Summary from {Path(summary_file_path_str).stem}:\n{msg['content']}\n\n"
                                break # Take first assistant message as the summary content
                    elif summary_data.get('content'): # Fallback for older/direct content format
                         combined_content += f"Summary from {Path(summary_file_path_str).stem}:\n{summary_data['content']}\n\n"

                except Exception as e:
                    print(f"Error loading conversation summary {summary_file_path_str}: {e}")
            
            if not combined_content.strip():
                print("Skipping STM creation due to empty combined content from batch.")
                continue

            # Format for summarizer (as a single user message asking to summarize)
            conversation_for_stm_summary = [{
                "role": "user",
                "content": f"Please create a concise meta-summary of the following conversation summaries. Combine their key insights, decisions, and outcomes into a single coherent narrative. Ensure all critical information is retained. \n\n{combined_content.strip()}"
            }]
            
            print(f"Generating STM summary for files: {batch_to_process}")
            try:
                # Use stream_summarize_conversation and collect chunks
                stm_summary_chunks = []
                for chunk in self.summarizer.stream_summarize_conversation(conversation_for_stm_summary):
                    stm_summary_chunks.append(chunk)
                
                stm_summary_text = "".join(stm_summary_chunks).strip()

                if not stm_summary_text:
                    print("STM summary generation resulted in empty text. Skipping save.")
                    continue

                saved_stm_path = self.logger._save_stm_summary(stm_summary_text, constituent_files_for_stm)
                print(f"Successfully created and saved STM: {saved_stm_path}")
            except Exception as e:
                print(f"Error during STM summary generation or saving: {e}")


    def _create_ltm_summaries_if_needed(self):
        """
        Checks for new STM summaries and processes them into LTMs.
        """
        print("Checking for new STM summaries to process into LTMs...")
        all_stm_summary_files = [str(Path(f)) for f in self.logger.get_stm_summary_files()]
        all_stm_summary_files.sort(key=lambda f: Path(f).stem, reverse=True) # Newest first for determining 'new'

        ltm_files = self.logger.get_ltm_summary_files()
        processed_stm_files = self._get_processed_files_from_meta_summaries(ltm_files, 'constituent_stms')
        
        new_stm_summary_files = [
            f for f in all_stm_summary_files if f not in processed_stm_files
        ]

        if not new_stm_summary_files:
            print("No new STM summaries to process for LTMs.")
            return

        print(f"Found {len(new_stm_summary_files)} new STM summaries for LTM creation.")
        
        # Reverse for batching so older ones are processed first
        new_stm_summary_files.reverse()

        for i in range(0, len(new_stm_summary_files), NUM_STM_FOR_LTM):
            batch_to_process = new_stm_summary_files[i:i + NUM_STM_FOR_LTM]
            if len(batch_to_process) < NUM_STM_FOR_LTM:
                print(f"Skipping LTM creation for batch of {len(batch_to_process)} STMs (less than {NUM_STM_FOR_LTM}). Will process later.")
                continue

            print(f"Processing batch of {len(batch_to_process)} STM summaries for LTM...")
            constituent_files_for_ltm = [str(Path(f)) for f in batch_to_process]

            combined_content = ""
            for stm_file_path_str in batch_to_process:
                try:
                    stm_data = self.logger.load_stm_summary_file(stm_file_path_str)
                    if stm_data and stm_data.get('content'):
                        combined_content += f"Short-Term Memory Snapshot ({Path(stm_file_path_str).stem}):\n{stm_data['content']}\n\n"
                except Exception as e:
                    print(f"Error loading STM summary {stm_file_path_str}: {e}")

            if not combined_content.strip():
                print("Skipping LTM creation due to empty combined content from STM batch.")
                continue
            
            conversation_for_ltm_summary = [{
                "role": "user",
                "content": f"Please create a high-level, condensed long-term memory synthesis from the following short-term memory snapshots. Identify overarching themes, critical long-term takeaways, and core knowledge. Distill this into a brief, potent summary. \n\n{combined_content.strip()}"
            }]

            print(f"Generating LTM summary for STM files: {batch_to_process}")
            try:
                # Use stream_summarize_conversation and collect chunks
                ltm_summary_chunks = []
                for chunk in self.summarizer.stream_summarize_conversation(conversation_for_ltm_summary):
                    ltm_summary_chunks.append(chunk)
                
                ltm_summary_text = "".join(ltm_summary_chunks).strip()

                if not ltm_summary_text:
                    print("LTM summary generation resulted in empty text. Skipping save.")
                    continue
                
                saved_ltm_path = self.logger._save_ltm_summary(ltm_summary_text, constituent_files_for_ltm)
                print(f"Successfully created and saved LTM: {saved_ltm_path}")
            except Exception as e:
                print(f"Error during LTM summary generation or saving: {e}") 