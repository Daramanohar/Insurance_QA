"""
Ollama Client for Answer Generation
This module handles communication with Ollama to generate answers using Mistral-7B
"""

import requests
import json
import logging
from typing import List, Dict, Optional
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Client for interacting with Ollama API to generate answers using LLM
    """
    
    def __init__(self, model: str = None, base_url: str = None):
        """
        Initialize Ollama client.
        
        Args:
            model: Model name (e.g., 'mistral:7b')
            base_url: Base URL for Ollama API
        """
        self.model = model or config.OLLAMA_MODEL
        self.base_url = base_url or config.OLLAMA_BASE_URL
        self.generate_url = f"{self.base_url}/api/generate"
        self.chat_url = f"{self.base_url}/api/chat"
        
    def check_connection(self) -> bool:
        """
        Check if Ollama is running and accessible.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            return False
    
    def generate_answer(
        self,
        question: str,
        context_docs: List[Dict],
        conversation_history: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """
        Generate an answer using retrieved context and conversation history.
        
        Args:
            question: User's question
            context_docs: List of similar Q&A pairs from Pinecone
            conversation_history: Previous conversation turns
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated answer string
        """
        # Build context from retrieved documents
        context = self._build_context(context_docs)
        
        # Build prompt with context and conversation history
        prompt = self._build_prompt(question, context, conversation_history)
        
        # Generate answer using Ollama
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=120  # Increased timeout for Mistral
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                return answer
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return "I apologize, but I'm having trouble generating a response right now."
                
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
            return "The request took too long. Please try again with a simpler question."
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I apologize, but I encountered an error while processing your question."
    
    def _build_context(self, context_docs: List[Dict]) -> str:
        """
        Build context string from retrieved documents.
        
        Args:
            context_docs: List of similar Q&A pairs
            
        Returns:
            Formatted context string
        """
        if not context_docs:
            return "No specific context found in the knowledge base."
        
        context_parts = []
        for i, doc in enumerate(context_docs[:3], 1):  # Use top 3 documents
            context_parts.append(
                f"Reference {i} (Relevance: {doc['score']:.2f}):\n"
                f"Q: {doc['question']}\n"
                f"A: {doc['answer']}\n"
            )
        
        return "\n".join(context_parts)
    
    def _build_prompt(
        self,
        question: str,
        context: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Build a comprehensive prompt for the LLM.
        
        Args:
            question: User's current question
            context: Retrieved context from knowledge base
            conversation_history: Previous conversation turns
            
        Returns:
            Formatted prompt string
        """
        # System instruction
        system_prompt = """You are an expert insurance advisor assistant. Your role is to provide accurate, helpful, and clear answers about insurance topics including health, auto, life, and other insurance types.

Instructions:
- Use the provided reference materials from the knowledge base to formulate your answer
- If the references don't fully answer the question, use your knowledge to provide additional helpful information
- Be concise but comprehensive
- Use clear, simple language that policyholders can understand
- If you're uncertain or the question is outside insurance topics, politely say so
- Don't make up information; stick to facts
"""
        
        # Build conversation context if available
        conversation_context = ""
        if conversation_history:
            conversation_context = "\n\nPrevious Conversation:\n"
            for turn in conversation_history[-3:]:  # Last 3 turns for context
                conversation_context += f"User: {turn['user']}\n"
                conversation_context += f"Assistant: {turn['assistant']}\n"
        
        # Combine all parts
        prompt = f"""{system_prompt}

Knowledge Base References:
{context}
{conversation_context}

Current Question: {question}

Answer: """
        
        return prompt
    
    def generate_simple_answer(self, question: str) -> str:
        """
        Generate a simple answer without context (fallback mode).
        
        Args:
            question: User's question
            
        Returns:
            Generated answer
        """
        prompt = f"""You are an insurance expert. Answer this question concisely:

Question: {question}

Answer: """
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 300
                }
            }
            
            response = requests.post(self.generate_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return "I couldn't generate an answer. Please try again."
                
        except Exception as e:
            logger.error(f"Error in simple answer generation: {e}")
            return "An error occurred while processing your question."


def test_ollama_connection():
    """
    Test function to verify Ollama connection and model availability
    """
    print("Testing Ollama connection...")
    client = OllamaClient()
    
    if client.check_connection():
        print("✓ Ollama is running and accessible")
        
        # Test generation
        print("\nTesting answer generation...")
        test_question = "What is a deductible?"
        test_context = [{
            'question': 'What is a deductible in insurance?',
            'answer': 'A deductible is the amount you pay out of pocket before your insurance coverage kicks in.',
            'score': 0.95
        }]
        
        answer = client.generate_answer(test_question, test_context)
        print(f"\nQuestion: {test_question}")
        print(f"Answer: {answer}")
    else:
        print("✗ Cannot connect to Ollama")
        print("\nMake sure Ollama is running:")
        print("  1. Install Ollama from https://ollama.ai")
        print("  2. Run: ollama pull mistral")
        print("  3. Ollama should be running on http://localhost:11434")


if __name__ == "__main__":
    test_ollama_connection()


