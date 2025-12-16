"""
Answer Validation Layer - STRICT QUALITY CONTROL
Ensures all responses are English-only, factually grounded, and professional
"""

import re
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class AnswerValidator:
    """
    Validates generated answers for quality, language, and factual accuracy
    """
    
    # Non-English character patterns (Tagalog, Spanish, etc.)
    NON_ENGLISH_PATTERNS = [
        r'[àáâãäåèéêëìíîïòóôõöùúûü]',  # Accented characters
        r'[ñÑ]',  # Spanish
        r'[\u0080-\u024F]',  # Extended Latin
        r'[\u1E00-\u1EFF]',  # Vietnamese
        r'[\u0100-\u017F]',  # Filipino/Tagalog extended
    ]
    
    # Prohibited phrases indicating hallucination
    HALLUCINATION_INDICATORS = [
        'based on the documents',
        'according to the reference',
        'the document states',
        'as mentioned in',
        'the provided context shows',
        'I cannot find',
        'I don\'t have information'
    ]
    
    # Insurance domain violations
    FACTUAL_VIOLATIONS = [
        'mileage.*no.*claim.*bonus',  # NCB doesn't depend on mileage
        'driving.*under.*influence.*legal',  # DUI is never legal
        'cancel.*anytime.*no.*penalty',  # Usually penalties exist
    ]
    
    def validate_answer(self, answer: str) -> Tuple[bool, Optional[str]]:
        """
        Validate answer for quality and correctness
        
        Args:
            answer: Generated answer text
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not answer or len(answer.strip()) < 10:
            return False, "Answer too short or empty"
        
        # Check 1: English-only enforcement
        is_english, lang_error = self._check_english_only(answer)
        if not is_english:
            logger.error(f"NON-ENGLISH DETECTED: {lang_error}")
            return False, f"Language violation: {lang_error}"
        
        # Check 2: No hallucination indicators
        has_hallucination = self._check_hallucinations(answer)
        if has_hallucination:
            logger.warning("Hallucination indicators detected")
            return False, "Answer contains hallucination markers"
        
        # Check 3: No factual violations
        has_violation, violation_msg = self._check_factual_violations(answer)
        if has_violation:
            logger.error(f"FACTUAL VIOLATION: {violation_msg}")
            return False, f"Factual error: {violation_msg}"
        
        # Check 4: Professional quality
        is_professional = self._check_professional_quality(answer)
        if not is_professional:
            logger.warning("Answer quality below professional standard")
            return False, "Answer quality insufficient"
        
        return True, None
    
    def _check_english_only(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if text is English-only
        
        Returns:
            Tuple of (is_english, error_message)
        """
        for pattern in self.NON_ENGLISH_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return False, f"Non-English characters detected: {matches[:5]}"
        
        # Check for common non-English words
        non_english_words = ['hindi', 'kung', 'ang', 'mga', 'sa', 'ng', 'po', 'ano', 'bakit']
        words = text.lower().split()
        found_foreign = [w for w in non_english_words if w in words]
        
        if found_foreign:
            return False, f"Non-English words: {found_foreign}"
        
        return True, None
    
    def _check_hallucinations(self, text: str) -> bool:
        """Check for hallucination indicators"""
        text_lower = text.lower()
        
        for indicator in self.HALLUCINATION_INDICATORS:
            if indicator.lower() in text_lower:
                return True
        
        return False
    
    def _check_factual_violations(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check for known factual violations in insurance domain
        
        Returns:
            Tuple of (has_violation, violation_description)
        """
        text_lower = text.lower()
        
        for pattern in self.FACTUAL_VIOLATIONS:
            if re.search(pattern, text_lower):
                return True, f"Detected incorrect claim matching pattern: {pattern}"
        
        return False, None
    
    def _check_professional_quality(self, text: str) -> bool:
        """
        Check if answer meets professional quality standards
        
        Returns:
            True if professional quality
        """
        # Must have reasonable length
        if len(text) < 50:
            return False
        
        # Should not be overly long (rambling)
        if len(text) > 2000:
            return False
        
        # Should have proper punctuation
        if text.count('.') == 0 and text.count('!') == 0:
            return False
        
        # Should not be all caps (shouting)
        if text.isupper():
            return False
        
        return True

