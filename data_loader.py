"""
Data Loader for InsuranceQA Dataset
This module handles loading, processing, and preparing the insurance Q&A dataset
"""

from datasets import load_dataset
import pandas as pd
from typing import List, Dict, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InsuranceDataLoader:
    """
    Loads and processes the InsuranceQA-v2 dataset from Hugging Face.
    The dataset contains insurance-related questions and expert answers.
    """
    
    def __init__(self, dataset_name: str = "deccan-ai/insuranceQA-v2"):
        """
        Initialize the data loader with dataset name.
        
        Args:
            dataset_name: Name of the dataset on Hugging Face
        """
        self.dataset_name = dataset_name
        self.dataset = None
        self.processed_data = []
        
    def load_dataset(self, split: str = "train") -> None:
        """
        Load the dataset from Hugging Face.
        
        Args:
            split: Dataset split to load ('train', 'test', or 'valid')
        """
        try:
            logger.info(f"Loading dataset: {self.dataset_name} ({split} split)")
            self.dataset = load_dataset(self.dataset_name, split=split)
            logger.info(f"Dataset loaded successfully with {len(self.dataset)} samples")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def process_dataset(self) -> List[Dict]:
        """
        Process the dataset into a format suitable for embedding and storage.
        
        The InsuranceQA dataset typically contains:
        - question: The insurance question
        - answers: List of possible answers
        - ground_truth: Index of correct answer(s)
        
        Returns:
            List of dictionaries containing processed Q&A pairs
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        logger.info("Processing dataset...")
        self.processed_data = []
        
        # Log first item structure for debugging
        if len(self.dataset) > 0:
            first_item = self.dataset[0]
            logger.info(f"Dataset keys: {list(first_item.keys())}")
            logger.info(f"First item sample: {str(first_item)[:200]}...")
        
        for idx, item in enumerate(self.dataset):
            try:
                # Try different possible field names
                question = (item.get('input') or 
                           item.get('question') or 
                           item.get('query') or 
                           item.get('text') or '')
                
                # Try different answer field names
                answers = (item.get('output') or 
                          item.get('answers') or 
                          item.get('answer') or 
                          item.get('response') or 
                          item.get('responses') or [])
                
                # Handle different answer formats
                if isinstance(answers, str):
                    answers = [answers]
                elif not isinstance(answers, list):
                    answers = []
                
                # Skip if no question or no answers
                if not question or not answers:
                    if idx < 5:  # Log first few skipped items
                        logger.warning(f"Skipping item {idx}: question='{question[:50] if question else 'None'}', answers count={len(answers)}")
                    continue
                
                # Get ground truth index if available
                ground_truth_idx = (item.get('ground_truth') or 
                                   item.get('correct_answer') or 
                                   item.get('label') or [])
                if isinstance(ground_truth_idx, int):
                    ground_truth_idx = [ground_truth_idx]
                
                # Create entries for each valid answer
                # If ground_truth exists, use only those answers; otherwise use first answer only
                if ground_truth_idx and len(ground_truth_idx) > 0:
                    answer_indices = [i for i in ground_truth_idx if i < len(answers)]
                else:
                    # Use only first answer to avoid duplicates
                    answer_indices = [0] if answers else []
                
                for ans_idx in answer_indices:
                    if ans_idx < len(answers):
                        answer = answers[ans_idx]
                        
                        # Skip empty answers
                        if not answer or (isinstance(answer, str) and len(answer.strip()) == 0):
                            continue
                        
                        # Create a combined text for better context
                        combined_text = f"Question: {question}\n\nAnswer: {answer}"
                        
                        # Classify domain for better retrieval filtering
                        domain = self._classify_domain(question, answer)
                        
                        self.processed_data.append({
                            'id': f"qa_{idx}_{ans_idx}",
                            'question': question,
                            'answer': answer,
                            'combined_text': combined_text,
                            'metadata': {
                                'source': 'insuranceQA-v2',
                                'original_idx': idx,
                                'domain': domain  # Add domain for Pinecone filtering
                            }
                        })
            
            except Exception as e:
                logger.warning(f"Error processing item {idx}: {e}")
                if idx < 5:  # Show detailed error for first few items
                    import traceback
                    logger.warning(traceback.format_exc())
                continue
        
        logger.info(f"Processed {len(self.processed_data)} Q&A pairs")
        return self.processed_data
    
    def _classify_domain(self, question: str, answer: str) -> str:
        """Classify insurance domain from question/answer text"""
        text = (question + ' ' + answer).lower()
        
        # Auto insurance keywords
        auto_keywords = ['auto', 'car', 'vehicle', 'driving', 'accident', 'collision', 'comprehensive', 
                        'liability', 'deductible', 'premium', 'claim', 'no-claim', 'ncb', 'u-haul', 
                        'uhaul', 'rental', 'truck', 'commercial vehicle', 'driving', 'road']
        if any(kw in text for kw in auto_keywords):
            return 'auto'
        
        # Health insurance keywords
        health_keywords = ['health', 'medicare', 'medigap', 'medical', 'prescription', 'doctor', 'hospital',
                          'surgery', 'procedure', 'coverage', 'plan', 'copay', 'rhinoplasty', 'duodenal',
                          'cpap', 'uti', 'weight loss']
        if any(kw in text for kw in health_keywords):
            return 'health'
        
        # Life insurance keywords
        life_keywords = ['life insurance', 'term life', 'whole life', 'endowment', 'death benefit', 
                        'beneficiary', 'policy', 'coverage', 'borrow', 'cash value']
        if any(kw in text for kw in life_keywords):
            return 'life'
        
        # Home insurance keywords
        home_keywords = ['home', 'homeowner', 'property', 'house', 'flood', 'fire', 'theft', 'damage',
                        'coverage', 'dog', 'pet', 'liability']
        if any(kw in text for kw in home_keywords):
            return 'home'
        
        return 'general'
    
    def get_processed_data(self) -> List[Dict]:
        """
        Get the processed data.
        
        Returns:
            List of processed Q&A pairs
        """
        if not self.processed_data:
            logger.warning("No processed data available. Call process_dataset() first.")
        return self.processed_data
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Convert processed data to pandas DataFrame for easier analysis.
        
        Returns:
            DataFrame with processed data
        """
        if not self.processed_data:
            raise ValueError("No processed data available.")
        
        return pd.DataFrame(self.processed_data)
    
    def save_to_csv(self, filepath: str = "insurance_qa_data.csv") -> None:
        """
        Save processed data to CSV file for backup/inspection.
        
        Args:
            filepath: Path to save the CSV file
        """
        df = self.get_dataframe()
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")


def main():
    """
    Test the data loader functionality
    """
    loader = InsuranceDataLoader()
    loader.load_dataset(split="train")
    data = loader.process_dataset()
    
    # Display sample data
    print(f"\nTotal Q&A pairs: {len(data)}")
    print("\nSample data:")
    for i, item in enumerate(data[:3]):
        print(f"\n--- Sample {i+1} ---")
        print(f"ID: {item['id']}")
        print(f"Question: {item['question'][:100]}...")
        print(f"Answer: {item['answer'][:100]}...")
    
    # Save to CSV
    loader.save_to_csv()


if __name__ == "__main__":
    main()


