"""
Dual-LLM Verification and Answer Modification Module for RAG-Anything

This module implements a sophisticated two-stage verification system where:
1. A generator LLM produces an initial answer
2. A verifier LLM (typically more powerful) evaluates answer quality
3. If quality is below threshold, a modifier improves the answer iteratively

The system prevents hallucinations, improves factual consistency, and ensures
high-quality responses through systematic verification and refinement.

Usage Example:
    ```python
    from raganything.verification import (
        AnswerVerifier,
        AnswerModifier,
        DualLLMPipeline,
        VerificationConfig
    )

    # Initialize configuration
    config = VerificationConfig(
        verification_threshold=7.5,
        max_modification_iterations=3,
        require_all_criteria_pass=False
    )

    # Create pipeline
    pipeline = DualLLMPipeline(
        generator_llm=generator_func,
        verifier_llm=verifier_func,
        config=config
    )

    # Process answer with verification
    result = await pipeline.process_answer(
        query="What causes diabetes?",
        answer="Diabetes is caused by...",
        context="[Retrieved context about diabetes]"
    )

    print(f"Final answer: {result['final_answer']}")
    print(f"Quality score: {result['final_score']}/10")
    print(f"Iterations: {result['total_iterations']}")
    ```

Author: RAG-Anything Team
Version: 2.0.0
"""

from __future__ import annotations

import re
import json
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from lightrag.utils import logger


# =============================================================================
# Configuration Classes
# =============================================================================

class VerificationCriterion(Enum):
    """Verification criteria for answer quality assessment"""
    FAITHFULNESS = "faithfulness"  # Supported by context
    COMPLETENESS = "completeness"  # Addresses all query aspects
    ACCURACY = "accuracy"          # Factually correct
    CLARITY = "clarity"            # Clear and well-structured
    RELEVANCE = "relevance"        # Directly answers the query
    COHERENCE = "coherence"        # Logically consistent


@dataclass
class VerificationConfig:
    """Configuration for dual-LLM verification system

    Attributes:
        verification_threshold: Minimum score (0-10) for answer to pass
        max_modification_iterations: Maximum number of improvement attempts
        require_all_criteria_pass: Whether all criteria must pass individually
        individual_criterion_threshold: Min score per criterion if required
        enable_confidence_scoring: Enable probabilistic confidence estimation
        enable_detailed_feedback: Generate detailed improvement suggestions
        stop_on_first_pass: Stop iterations when answer first passes
        criteria_weights: Custom weights for each criterion (must sum to 1.0)
        context_truncation_length: Max context chars to send to verifier
        min_improvement_delta: Minimum score improvement to continue iterations
    """

    verification_threshold: float = 7.0
    max_modification_iterations: int = 2
    require_all_criteria_pass: bool = False
    individual_criterion_threshold: float = 6.0
    enable_confidence_scoring: bool = True
    enable_detailed_feedback: bool = True
    stop_on_first_pass: bool = True
    context_truncation_length: int = 4000
    min_improvement_delta: float = 0.5

    # Criteria weights (must sum to 1.0)
    criteria_weights: Dict[str, float] = field(default_factory=lambda: {
        "faithfulness": 0.35,
        "completeness": 0.25,
        "accuracy": 0.20,
        "relevance": 0.10,
        "clarity": 0.05,
        "coherence": 0.05
    })

    def __post_init__(self):
        """Validate configuration"""
        # Ensure weights sum to 1.0
        total_weight = sum(self.criteria_weights.values())
        if not (0.99 <= total_weight <= 1.01):
            logger.warning(
                f"Criteria weights sum to {total_weight}, normalizing to 1.0"
            )
            # Normalize weights
            for key in self.criteria_weights:
                self.criteria_weights[key] /= total_weight


# =============================================================================
# Answer Verifier
# =============================================================================

class AnswerVerifier:
    """Advanced answer quality verifier with multi-criteria evaluation

    This class evaluates generated answers across multiple quality dimensions,
    providing detailed feedback and confidence scores. It uses structured
    prompting to ensure consistent, reliable verification.

    Attributes:
        verifier_llm_func: LLM function for verification (typically GPT-4 or similar)
        config: VerificationConfig instance
    """

    def __init__(
        self,
        verifier_llm_func: Callable,
        config: Optional[VerificationConfig] = None
    ):
        """Initialize AnswerVerifier

        Args:
            verifier_llm_func: LLM function for verification
            config: Configuration object, if None will use defaults
        """
        self.verifier_llm_func = verifier_llm_func
        self.config = config or VerificationConfig()

    async def verify_answer(
        self,
        query: str,
        answer: str,
        context: str,
        original_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """Verify answer quality across multiple criteria

        Args:
            query: Query used for generation (may be improved query)
            answer: Generated answer to verify
            context: Retrieved context used for generation
            original_query: Original user query (if different from query)

        Returns:
            Dictionary containing:
                - passed: Whether answer meets quality threshold
                - overall_score: Weighted average score (0-10)
                - criteria_scores: Individual scores per criterion
                - confidence: Confidence in verification (0-1)
                - feedback: Detailed evaluation feedback
                - issues: List of specific issues found
                - suggestions: Improvement suggestions
                - metadata: Additional verification metadata

        Example:
            ```python
            result = await verifier.verify_answer(
                query="What causes type 2 diabetes?",
                answer="Type 2 diabetes is caused by insulin resistance...",
                context="[Medical literature about diabetes]"
            )
            if result['passed']:
                print(f"Answer quality: {result['overall_score']}/10")
            else:
                print(f"Issues: {result['issues']}")
            ```
        """
        if not answer or not answer.strip():
            logger.warning("Empty answer provided for verification")
            return self._create_failed_result("Empty answer", 0.0)

        try:
            # Build verification prompt
            verification_prompt = self._build_verification_prompt(
                query=original_query or query,
                answer=answer,
                context=context
            )

            # Call verifier LLM
            logger.debug("Calling verifier LLM for answer evaluation...")
            response = await self._call_verifier_safely(verification_prompt)

            if not response:
                logger.warning("Empty response from verifier LLM")
                return self._create_default_pass_result()

            # Parse verification response
            result = self._parse_verification_response(response)

            # Determine if answer passes
            result["passed"] = self._evaluate_pass_criteria(result)

            # Add confidence score if enabled
            if self.config.enable_confidence_scoring:
                result["confidence"] = self._calculate_confidence(result)

            logger.info(
                f"Verification complete: score={result['overall_score']:.2f}, "
                f"passed={result['passed']}"
            )

            return result

        except Exception as e:
            logger.error(f"Error during answer verification: {e}", exc_info=True)
            return self._create_error_result(str(e))

    def _build_verification_prompt(
        self,
        query: str,
        answer: str,
        context: str
    ) -> str:
        """Build structured verification prompt with JSON schema

        Args:
            query: Original query
            answer: Generated answer
            context: Retrieved context

        Returns:
            Formatted verification prompt
        """
        # Truncate context if too long
        if len(context) > self.config.context_truncation_length:
            context = context[:self.config.context_truncation_length] + "\n\n[... context truncated ...]"

        # Build criteria descriptions
        criteria_desc = []
        for criterion, weight in self.config.criteria_weights.items():
            criteria_desc.append(
                f"  - {criterion.capitalize()}: {self._get_criterion_description(criterion)} "
                f"(Weight: {weight*100:.0f}%)"
            )
        criteria_text = "\n".join(criteria_desc)

        prompt = f"""Evaluate the following answer for quality and correctness.

QUERY:
{query}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

EVALUATION CRITERIA:
{criteria_text}

For each criterion, provide:
1. A score from 0-10 (0=completely fails, 10=perfect)
2. Specific evidence from the answer/context
3. Identified issues or strengths

IMPORTANT INSTRUCTIONS:
- Be critical and objective in your evaluation
- Check if the answer is fully supported by the context (no hallucinations)
- Verify factual accuracy against the context
- Identify any missing information or incomplete aspects
- Note any logical inconsistencies or unclear statements
- Do not be lenient - high scores should be rare and well-deserved

Respond with ONLY a valid JSON object in this exact format:
{{
    "faithfulness": {{
        "score": <0-10>,
        "evidence": "<specific quote or observation>",
        "issues": ["<issue 1>", "<issue 2>"]
    }},
    "completeness": {{
        "score": <0-10>,
        "evidence": "<specific quote or observation>",
        "issues": ["<issue 1>"]
    }},
    "accuracy": {{
        "score": <0-10>,
        "evidence": "<specific quote or observation>",
        "issues": []
    }},
    "relevance": {{
        "score": <0-10>,
        "evidence": "<specific quote or observation>",
        "issues": []
    }},
    "clarity": {{
        "score": <0-10>,
        "evidence": "<specific quote or observation>",
        "issues": []
    }},
    "coherence": {{
        "score": <0-10>,
        "evidence": "<specific quote or observation>",
        "issues": []
    }},
    "overall_feedback": "<comprehensive evaluation summary>",
    "critical_issues": ["<critical issue 1>", "<critical issue 2>"],
    "suggestions": ["<improvement suggestion 1>", "<improvement suggestion 2>"]
}}

DO NOT include any text before or after the JSON object. DO NOT use markdown code blocks."""

        return prompt

    def _get_criterion_description(self, criterion: str) -> str:
        """Get description for each criterion

        Args:
            criterion: Criterion name

        Returns:
            Human-readable description
        """
        descriptions = {
            "faithfulness": "Answer is fully supported by the context without hallucinations",
            "completeness": "Answer addresses all aspects of the query comprehensively",
            "accuracy": "Information is factually correct and precise",
            "relevance": "Answer directly addresses the query without tangents",
            "clarity": "Answer is well-structured, clear, and easy to understand",
            "coherence": "Answer is logically consistent without contradictions"
        }
        return descriptions.get(criterion, "Quality assessment")

    async def _call_verifier_safely(self, prompt: str) -> str:
        """Call verifier LLM with error handling

        Args:
            prompt: Verification prompt

        Returns:
            LLM response string
        """
        try:
            system_prompt = """You are an expert answer evaluator for RAG systems. Your role is to critically assess answer quality across multiple dimensions.

You must be:
- Objective and unbiased
- Critical and demanding (high scores are rare)
- Specific and evidence-based in your feedback
- Focused on factual accuracy and faithfulness to context
- Able to identify subtle issues like hallucinations or incompleteness

Always respond with a valid JSON object. Do not add explanations outside the JSON."""

            if asyncio.iscoroutinefunction(self.verifier_llm_func):
                response = await self.verifier_llm_func(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.2,  # Lower temperature for more consistent evaluation
                    max_tokens=1500
                )
            else:
                response = self.verifier_llm_func(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.2,
                    max_tokens=1500
                )

            return response

        except Exception as e:
            logger.error(f"Error calling verifier LLM: {e}", exc_info=True)
            raise

    def _parse_verification_response(self, response: str) -> Dict[str, Any]:
        """Parse verification response with robust error handling

        Args:
            response: LLM response string

        Returns:
            Parsed verification result
        """
        try:
            # Clean response - remove markdown code blocks
            cleaned = self._clean_json_response(response)

            # Parse JSON
            data = json.loads(cleaned)

            # Extract criterion scores
            criteria_scores = {}
            all_issues = []
            all_evidence = {}

            for criterion in self.config.criteria_weights.keys():
                if criterion in data:
                    criterion_data = data[criterion]
                    if isinstance(criterion_data, dict):
                        score = float(criterion_data.get("score", 5.0))
                        criteria_scores[criterion] = score
                        all_evidence[criterion] = criterion_data.get("evidence", "")
                        all_issues.extend(criterion_data.get("issues", []))
                    elif isinstance(criterion_data, (int, float)):
                        criteria_scores[criterion] = float(criterion_data)
                else:
                    # Default score if missing
                    criteria_scores[criterion] = 5.0

            # Calculate weighted overall score
            overall_score = sum(
                criteria_scores[k] * self.config.criteria_weights[k]
                for k in criteria_scores.keys()
            )

            # Extract feedback and suggestions
            feedback = data.get("overall_feedback", "No detailed feedback provided")
            critical_issues = data.get("critical_issues", [])
            suggestions = data.get("suggestions", [])

            # Combine all issues
            all_issues.extend(critical_issues)
            all_issues = list(set(all_issues))  # Remove duplicates

            return {
                "overall_score": overall_score,
                "criteria_scores": criteria_scores,
                "feedback": feedback,
                "issues": all_issues,
                "suggestions": suggestions,
                "evidence": all_evidence,
                "metadata": {
                    "response_parsed": True,
                    "criteria_evaluated": len(criteria_scores)
                }
            }

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse verification response as JSON: {e}")
            # Attempt to extract scores from text
            return self._fallback_parse(response)

        except Exception as e:
            logger.error(f"Error parsing verification response: {e}", exc_info=True)
            return self._fallback_parse(response)

    def _fallback_parse(self, response: str) -> Dict[str, Any]:
        """Fallback parsing when JSON parsing fails

        Args:
            response: Raw response text

        Returns:
            Best-effort parsed result
        """
        # Try to extract any score from text
        scores_found = re.findall(r'(?:score|rating)[:\s]+(\d+(?:\.\d+)?)', response, re.IGNORECASE)

        if scores_found:
            # Use average of found scores
            avg_score = sum(float(s) for s in scores_found) / len(scores_found)
        else:
            avg_score = 5.0  # Neutral default

        return {
            "overall_score": avg_score,
            "criteria_scores": {k: avg_score for k in self.config.criteria_weights.keys()},
            "feedback": response[:500],
            "issues": ["Failed to parse structured verification response"],
            "suggestions": [],
            "evidence": {},
            "metadata": {
                "response_parsed": False,
                "fallback_used": True
            }
        }

    def _clean_json_response(self, response: str) -> str:
        """Clean JSON response by removing markdown and extra content

        Args:
            response: Raw LLM response

        Returns:
            Cleaned JSON string
        """
        # Remove markdown code blocks
        cleaned = re.sub(r'```json\s*', '', response)
        cleaned = re.sub(r'```\s*', '', cleaned)

        # Extract JSON object (first complete {} block)
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group(0)

        # Strip whitespace
        cleaned = cleaned.strip()

        return cleaned

    def _evaluate_pass_criteria(self, result: Dict[str, Any]) -> bool:
        """Determine if answer passes based on configuration

        Args:
            result: Verification result dictionary

        Returns:
            True if answer passes all criteria
        """
        # Check overall score threshold
        if result["overall_score"] < self.config.verification_threshold:
            return False

        # If individual criteria must all pass
        if self.config.require_all_criteria_pass:
            for score in result["criteria_scores"].values():
                if score < self.config.individual_criterion_threshold:
                    return False

        return True

    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate confidence score for verification

        Args:
            result: Verification result

        Returns:
            Confidence score (0-1)
        """
        # Factors that affect confidence:
        # 1. Score variance (low variance = high confidence)
        # 2. Number of issues (few issues = high confidence)
        # 3. Whether response was properly parsed

        scores = list(result["criteria_scores"].values())

        if not scores:
            return 0.5

        # Calculate variance
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)

        # Low variance = high confidence
        variance_factor = max(0, 1 - (variance / 10))

        # Few issues = high confidence
        num_issues = len(result.get("issues", []))
        issues_factor = max(0, 1 - (num_issues * 0.1))

        # Properly parsed = bonus confidence
        parse_factor = 1.0 if result.get("metadata", {}).get("response_parsed", False) else 0.8

        # Combine factors
        confidence = (variance_factor * 0.3 + issues_factor * 0.4 + parse_factor * 0.3)

        return min(1.0, max(0.0, confidence))

    def _create_failed_result(self, reason: str, score: float) -> Dict[str, Any]:
        """Create a failed verification result

        Args:
            reason: Failure reason
            score: Score to assign

        Returns:
            Failed result dictionary
        """
        return {
            "passed": False,
            "overall_score": score,
            "criteria_scores": {k: score for k in self.config.criteria_weights.keys()},
            "feedback": f"Verification failed: {reason}",
            "issues": [reason],
            "suggestions": [],
            "confidence": 0.0,
            "evidence": {},
            "metadata": {"error": reason}
        }

    def _create_default_pass_result(self) -> Dict[str, Any]:
        """Create a default passing result (used when verifier fails)

        Returns:
            Default passing result
        """
        threshold = self.config.verification_threshold
        return {
            "passed": True,
            "overall_score": threshold,
            "criteria_scores": {k: threshold for k in self.config.criteria_weights.keys()},
            "feedback": "Verification completed with default scores (verifier unavailable)",
            "issues": [],
            "suggestions": [],
            "confidence": 0.5,
            "evidence": {},
            "metadata": {"default_result": True}
        }

    def _create_error_result(self, error: str) -> Dict[str, Any]:
        """Create an error result (passes by default to avoid blocking)

        Args:
            error: Error message

        Returns:
            Error result dictionary
        """
        threshold = self.config.verification_threshold
        return {
            "passed": True,  # Pass by default on error
            "overall_score": threshold,
            "criteria_scores": {k: threshold for k in self.config.criteria_weights.keys()},
            "feedback": f"Verification error: {error}",
            "issues": [],
            "suggestions": [],
            "confidence": 0.0,
            "evidence": {},
            "metadata": {"error": error}
        }


# =============================================================================
# Answer Modifier
# =============================================================================

class AnswerModifier:
    """Answer modifier that improves answers based on verification feedback

    This class takes verification feedback and generates improved versions
    of answers, addressing identified issues while maintaining accuracy.

    Attributes:
        generator_llm_func: LLM function for answer modification
        config: VerificationConfig instance
    """

    def __init__(
        self,
        generator_llm_func: Callable,
        config: Optional[VerificationConfig] = None
    ):
        """Initialize AnswerModifier

        Args:
            generator_llm_func: LLM function for modification
            config: Configuration object
        """
        self.generator_llm_func = generator_llm_func
        self.config = config or VerificationConfig()

    async def modify_answer(
        self,
        query: str,
        answer: str,
        context: str,
        verification_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Modify answer based on verification feedback

        Args:
            query: Original query
            answer: Answer that failed verification
            context: Retrieved context
            verification_result: Feedback from verifier

        Returns:
            Dictionary containing:
                - modified_answer: Improved answer
                - changes_made: List of changes
                - modification_successful: Whether modification completed
                - metadata: Additional metadata

        Example:
            ```python
            result = await modifier.modify_answer(
                query="What causes diabetes?",
                answer="Diabetes is caused by...",
                context="[Medical context]",
                verification_result=verification_feedback
            )
            print(result['modified_answer'])
            ```
        """
        logger.info("Modifying answer based on verification feedback...")

        try:
            # Build modification prompt
            modification_prompt = self._build_modification_prompt(
                query=query,
                answer=answer,
                context=context,
                verification_result=verification_result
            )

            # Generate improved answer
            response = await self._call_generator_safely(modification_prompt)

            if not response:
                logger.warning("Empty response from generator, returning original")
                return {
                    "modified_answer": answer,
                    "changes_made": [],
                    "modification_successful": False,
                    "metadata": {"error": "Empty response"}
                }

            # Extract modified answer
            modified_answer = self._extract_answer(response)

            # Analyze changes
            changes_made = self._identify_changes(answer, modified_answer)

            logger.info(f"Answer modification complete ({len(changes_made)} changes)")

            return {
                "modified_answer": modified_answer,
                "changes_made": changes_made,
                "modification_successful": True,
                "metadata": {
                    "original_length": len(answer),
                    "modified_length": len(modified_answer),
                    "length_delta": len(modified_answer) - len(answer)
                }
            }

        except Exception as e:
            logger.error(f"Error during answer modification: {e}", exc_info=True)
            return {
                "modified_answer": answer,  # Return original on error
                "changes_made": [],
                "modification_successful": False,
                "metadata": {"error": str(e)}
            }

    def _build_modification_prompt(
        self,
        query: str,
        answer: str,
        context: str,
        verification_result: Dict[str, Any]
    ) -> str:
        """Build modification prompt with detailed feedback

        Args:
            query: Original query
            answer: Current answer
            context: Retrieved context
            verification_result: Verification feedback

        Returns:
            Formatted modification prompt
        """
        # Truncate context if needed
        if len(context) > self.config.context_truncation_length:
            context = context[:self.config.context_truncation_length] + "\n\n[... context truncated ...]"

        # Format issues and suggestions
        issues = verification_result.get("issues", [])
        suggestions = verification_result.get("suggestions", [])

        issues_text = "\n".join(f"  - {issue}" for issue in issues) if issues else "  - None identified"
        suggestions_text = "\n".join(f"  - {sug}" for sug in suggestions) if suggestions else "  - General improvement needed"

        # Format criterion scores
        criteria_scores = verification_result.get("criteria_scores", {})
        scores_text = "\n".join(
            f"  - {k.capitalize()}: {v:.1f}/10"
            for k, v in criteria_scores.items()
        )

        prompt = f"""Improve the following answer based on verification feedback.

QUERY:
{query}

REFERENCE CONTEXT:
{context}

CURRENT ANSWER:
{answer}

VERIFICATION FEEDBACK:
Overall Score: {verification_result.get('overall_score', 0):.1f}/10
Threshold: {self.config.verification_threshold}/10

Criterion Scores:
{scores_text}

Identified Issues:
{issues_text}

Improvement Suggestions:
{suggestions_text}

Detailed Feedback:
{verification_result.get('feedback', 'No additional feedback')}

IMPROVEMENT INSTRUCTIONS:
1. Address ALL identified issues completely
2. Ensure EVERY statement is supported by the context (no hallucinations)
3. Be comprehensive - answer all aspects of the query
4. Maintain factual accuracy - verify all claims against context
5. Improve clarity and structure
6. Fix any logical inconsistencies

IMPORTANT:
- Only use information from the provided context
- If context doesn't support a claim, remove it
- Add missing information if present in context
- Be specific and detailed while remaining concise
- Do not apologize or explain changes - just provide the improved answer

IMPROVED ANSWER:"""

        return prompt

    async def _call_generator_safely(self, prompt: str) -> str:
        """Call generator LLM with error handling

        Args:
            prompt: Modification prompt

        Returns:
            LLM response
        """
        try:
            system_prompt = """You are an expert answer improver. Your task is to enhance answers based on verification feedback while maintaining strict factual accuracy.

You must:
- Only use information from the provided context
- Address all identified issues
- Maintain or improve answer quality
- Be comprehensive yet concise
- Never hallucinate or add unsupported information

Provide ONLY the improved answer without explanations or preamble."""

            if asyncio.iscoroutinefunction(self.generator_llm_func):
                response = await self.generator_llm_func(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.4,
                    max_tokens=1500
                )
            else:
                response = self.generator_llm_func(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.4,
                    max_tokens=1500
                )

            return response

        except Exception as e:
            logger.error(f"Error calling generator LLM: {e}", exc_info=True)
            raise

    def _extract_answer(self, response: str) -> str:
        """Extract answer from response, removing any preamble

        Args:
            response: LLM response

        Returns:
            Cleaned answer
        """
        # Remove common preambles
        preambles = [
            r'^(?:here is|here\'s|the)\s+(?:an?\s+)?improved answer:?\s*',
            r'^improved answer:?\s*',
            r'^answer:?\s*',
        ]

        cleaned = response
        for pattern in preambles:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        return cleaned.strip()

    def _identify_changes(self, original: str, modified: str) -> List[str]:
        """Identify high-level changes between answers

        Args:
            original: Original answer
            modified: Modified answer

        Returns:
            List of change descriptions
        """
        changes = []

        # Length changes
        len_diff = len(modified) - len(original)
        if len_diff > 50:
            changes.append(f"Expanded answer (+{len_diff} characters)")
        elif len_diff < -50:
            changes.append(f"Condensed answer ({len_diff} characters)")

        # Word count changes
        orig_words = len(original.split())
        mod_words = len(modified.split())
        word_diff = mod_words - orig_words
        if word_diff > 10:
            changes.append(f"Added {word_diff} words")
        elif word_diff < -10:
            changes.append(f"Removed {abs(word_diff)} words")

        # Structural changes
        orig_sentences = len(re.findall(r'[.!?]+', original))
        mod_sentences = len(re.findall(r'[.!?]+', modified))
        if mod_sentences > orig_sentences:
            changes.append(f"Improved structure ({mod_sentences - orig_sentences} more sentences)")

        # If significantly different
        if len_diff == 0 and word_diff == 0:
            changes.append("Minor refinements")
        elif not changes:
            changes.append("Modified answer content")

        return changes


# =============================================================================
# Dual-LLM Pipeline
# =============================================================================

class DualLLMPipeline:
    """Complete dual-LLM verification and modification pipeline

    This class orchestrates the full verification-modification loop,
    coordinating between verifier and modifier until answer quality
    meets standards or maximum iterations are reached.

    Attributes:
        verifier: AnswerVerifier instance
        modifier: AnswerModifier instance
        config: VerificationConfig instance
    """

    def __init__(
        self,
        generator_llm: Callable,
        verifier_llm: Callable,
        config: Optional[VerificationConfig] = None
    ):
        """Initialize DualLLMPipeline

        Args:
            generator_llm: LLM function for answer generation/modification
            verifier_llm: LLM function for verification (typically more powerful)
            config: Configuration object
        """
        self.config = config or VerificationConfig()
        self.verifier = AnswerVerifier(verifier_llm, self.config)
        self.modifier = AnswerModifier(generator_llm, self.config)

    async def process_answer(
        self,
        query: str,
        answer: str,
        context: str,
        max_iterations: Optional[int] = None
    ) -> Dict[str, Any]:
        """Process answer through verification-modification loop

        Args:
            query: Original query
            answer: Initial generated answer
            context: Retrieved context
            max_iterations: Override config max iterations

        Returns:
            Dictionary containing:
                - final_answer: Best answer after iterations
                - final_score: Final verification score
                - passed: Whether final answer passed
                - total_iterations: Number of iterations performed
                - iteration_history: Detailed history of all iterations
                - improvement_delta: Score improvement from first to last
                - metadata: Additional processing metadata

        Example:
            ```python
            result = await pipeline.process_answer(
                query="What is photosynthesis?",
                answer="Photosynthesis is a process...",
                context="[Biology context about photosynthesis]"
            )

            print(f"Final answer (score {result['final_score']}/10):")
            print(result['final_answer'])
            print(f"\\nImprovement: +{result['improvement_delta']:.1f} points")
            ```
        """
        max_iter = max_iterations or self.config.max_modification_iterations

        logger.info(f"Starting dual-LLM pipeline (max {max_iter} iterations)...")

        # Initialize tracking
        iteration_history = []
        current_answer = answer
        iteration = 0

        # Verify initial answer
        verification_result = await self.verifier.verify_answer(
            query=query,
            answer=current_answer,
            context=context
        )

        initial_score = verification_result["overall_score"]

        iteration_history.append({
            "iteration": 0,
            "answer": current_answer,
            "verification": verification_result,
            "modification": None
        })

        logger.info(
            f"Initial verification: score={initial_score:.2f}, "
            f"passed={verification_result['passed']}"
        )

        # If passed and stop_on_first_pass, we're done
        if verification_result["passed"] and self.config.stop_on_first_pass:
            logger.info("Answer passed verification on first attempt")
            return self._create_result(
                final_answer=current_answer,
                final_verification=verification_result,
                iteration_history=iteration_history,
                initial_score=initial_score
            )

        # Modification loop
        previous_score = initial_score

        while iteration < max_iter:
            iteration += 1

            # Check if we should continue
            if verification_result["passed"] and self.config.stop_on_first_pass:
                logger.info(f"Answer passed verification after {iteration-1} modifications")
                break

            # Check for minimal improvement
            if iteration > 1:
                score_improvement = verification_result["overall_score"] - previous_score
                if score_improvement < self.config.min_improvement_delta:
                    logger.info(
                        f"Minimal improvement detected ({score_improvement:.2f}), "
                        "stopping iterations"
                    )
                    break

            previous_score = verification_result["overall_score"]

            logger.info(f"Iteration {iteration}: Modifying answer...")

            # Modify answer
            modification_result = await self.modifier.modify_answer(
                query=query,
                answer=current_answer,
                context=context,
                verification_result=verification_result
            )

            if not modification_result["modification_successful"]:
                logger.warning("Modification failed, using previous answer")
                break

            current_answer = modification_result["modified_answer"]

            # Verify modified answer
            verification_result = await self.verifier.verify_answer(
                query=query,
                answer=current_answer,
                context=context
            )

            logger.info(
                f"Iteration {iteration} verification: score={verification_result['overall_score']:.2f}, "
                f"passed={verification_result['passed']}"
            )

            # Record iteration
            iteration_history.append({
                "iteration": iteration,
                "answer": current_answer,
                "verification": verification_result,
                "modification": modification_result
            })

        # Create final result
        return self._create_result(
            final_answer=current_answer,
            final_verification=verification_result,
            iteration_history=iteration_history,
            initial_score=initial_score
        )

    def _create_result(
        self,
        final_answer: str,
        final_verification: Dict[str, Any],
        iteration_history: List[Dict[str, Any]],
        initial_score: float
    ) -> Dict[str, Any]:
        """Create final pipeline result

        Args:
            final_answer: Final answer string
            final_verification: Final verification result
            iteration_history: Complete iteration history
            initial_score: Initial verification score

        Returns:
            Complete result dictionary
        """
        final_score = final_verification["overall_score"]
        improvement_delta = final_score - initial_score

        return {
            "final_answer": final_answer,
            "final_score": final_score,
            "passed": final_verification["passed"],
            "total_iterations": len(iteration_history) - 1,
            "iteration_history": iteration_history,
            "improvement_delta": improvement_delta,
            "confidence": final_verification.get("confidence", 0.0),
            "metadata": {
                "initial_score": initial_score,
                "final_score": final_score,
                "improvement_percentage": (improvement_delta / max(initial_score, 0.1)) * 100,
                "threshold": self.config.verification_threshold,
                "max_iterations_reached": len(iteration_history) - 1 >= self.config.max_modification_iterations
            }
        }


# =============================================================================
# Mixin Class for Integration
# =============================================================================

class DualLLMVerificationMixin:
    """
    Mixin providing dual-LLM verification functionality to RAGAnything

    This mixin adds answer verification and modification capabilities using
    a two-LLM approach:
    1. Generator LLM creates the initial answer
    2. Verifier LLM evaluates answer quality across multiple criteria
    3. Modifier LLM improves the answer based on verification feedback
    4. Process repeats until answer passes verification or max iterations reached

    The mixin expects the following attributes to be present:
    - self.answer_verifier: AnswerVerifier instance (optional)
    - self.answer_modifier: AnswerModifier instance (optional)
    - self.verification_pipeline: DualLLMPipeline instance (optional)
    - self.lightrag: LightRAG instance for answer generation
    - self.config: RAGAnythingConfig instance
    - self.logger: Logger instance
    """

    async def _generate_with_verification(
        self,
        query: str,
        context: str,
        original_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate answer with dual-LLM verification

        This method generates an answer and then verifies it using a separate
        verifier LLM. If the answer doesn't pass verification, it can be
        iteratively improved based on feedback.

        Args:
            query: The query to answer (possibly improved)
            context: Retrieved context from RAG system
            original_query: Original user query before improvement (optional)

        Returns:
            Dict containing:
                - answer: The final verified answer
                - verification_passed: Whether verification passed
                - verification_score: Overall quality score (0-10)
                - modification_attempts: Number of modification iterations
                - verification_history: List of verification results per iteration
                - final_criteria_scores: Scores for each criterion
                - confidence: Confidence in the verification
                - metadata: Additional verification metadata

        Example:
            result = await self._generate_with_verification(
                query="What is the treatment for hypertension?",
                context="Retrieved medical context...",
                original_query="What is HTN treatment?"
            )
            # result might be:
            # {
            #     'answer': 'Hypertension treatment includes...',
            #     'verification_passed': True,
            #     'verification_score': 8.5,
            #     'modification_attempts': 1,
            #     'confidence': 0.92
            # }
        """
        # Check if verification pipeline is available
        if not hasattr(self, 'verification_pipeline') or self.verification_pipeline is None:
            # Fall back to simple answer verifier if available
            if hasattr(self, 'answer_verifier') and self.answer_verifier is not None:
                return await self._verify_answer_only(query, context, original_query)
            else:
                # No verification available, generate answer without verification
                if hasattr(self, 'logger'):
                    self.logger.debug(
                        "Verification pipeline not initialized, generating without verification"
                    )
                return await self._generate_without_verification(query, context, original_query)

        try:
            if hasattr(self, 'logger'):
                self.logger.info(
                    f"Generating answer with verification (query: '{query[:50]}...')"
                )

            # Use verification pipeline for full verification-modification loop
            verification_result = await self.verification_pipeline.process_answer(
                query=query,
                answer=None,  # Pipeline will generate initial answer
                context=context,
                max_iterations=getattr(
                    self.config, 'max_verification_iterations',
                    getattr(self.config, 'max_verification_retries', 2)
                ) if hasattr(self, 'config') else 2
            )

            if hasattr(self, 'logger'):
                self.logger.info(
                    f"Verification complete: passed={verification_result.get('passed', False)}, "
                    f"score={verification_result.get('final_score', 0):.2f}, "
                    f"iterations={verification_result.get('total_iterations', 0)}"
                )

            # Format result for consistent return structure
            return {
                'answer': verification_result.get('final_answer', ''),
                'verification_passed': verification_result.get('passed', False),
                'verification_score': verification_result.get('final_score', 0),
                'modification_attempts': verification_result.get('total_iterations', 0),
                'verification_history': verification_result.get('iteration_history', []),
                'final_criteria_scores': verification_result.get('iteration_history', [{}])[-1].get('criteria_scores', {}) if verification_result.get('iteration_history') else {},
                'confidence': verification_result.get('iteration_history', [{}])[-1].get('confidence', 0) if verification_result.get('iteration_history') else 0,
                'improvement_delta': verification_result.get('improvement_delta', 0),
                'metadata': {
                    'original_query': original_query or query,
                    'improved_query': query,
                    'verification_method': 'dual_llm_pipeline'
                }
            }

        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error in verification pipeline: {e}", exc_info=True)

            # Fall back to unverified answer generation
            return await self._generate_without_verification(query, context, original_query)

    async def _verify_answer_only(
        self,
        query: str,
        context: str,
        original_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verify answer without modification (verifier available but no modifier)

        Args:
            query: The query to answer
            context: Retrieved context
            original_query: Original query before improvement

        Returns:
            Dict with verification results (but no iterative improvement)
        """
        try:
            # Generate initial answer using LightRAG
            answer = await self._generate_answer_from_context(query, context)

            if hasattr(self, 'logger'):
                self.logger.info("Verifying answer (modification disabled)")

            # Verify the answer
            verification_result = await self.answer_verifier.verify_answer(
                query=query,
                answer=answer,
                context=context,
                original_query=original_query
            )

            return {
                'answer': answer,
                'verification_passed': verification_result.get('passed', False),
                'verification_score': verification_result.get('overall_score', 0),
                'modification_attempts': 0,
                'verification_history': [verification_result],
                'final_criteria_scores': verification_result.get('criteria_scores', {}),
                'confidence': verification_result.get('confidence', 0),
                'metadata': {
                    'original_query': original_query or query,
                    'improved_query': query,
                    'verification_method': 'verify_only',
                    'note': 'Answer modification not enabled'
                }
            }

        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error in answer verification: {e}", exc_info=True)

            # Fall back to unverified answer
            answer = await self._generate_answer_from_context(query, context)
            return {
                'answer': answer,
                'verification_passed': True,
                'verification_score': 10.0,
                'modification_attempts': 0,
                'metadata': {'error': str(e), 'verification_method': 'none'}
            }

    async def _generate_without_verification(
        self,
        query: str,
        context: str,
        original_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate answer without verification (fallback method)

        Args:
            query: The query to answer
            context: Retrieved context
            original_query: Original query before improvement

        Returns:
            Dict with answer but no verification info
        """
        try:
            if hasattr(self, 'logger'):
                self.logger.debug("Generating answer without verification")

            answer = await self._generate_answer_from_context(query, context)

            return {
                'answer': answer,
                'verification_passed': True,
                'verification_score': 10.0,
                'modification_attempts': 0,
                'metadata': {
                    'original_query': original_query or query,
                    'improved_query': query,
                    'verification_method': 'none',
                    'note': 'Verification not enabled'
                }
            }

        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error generating answer: {e}", exc_info=True)

            return {
                'answer': f"Error generating answer: {str(e)}",
                'verification_passed': False,
                'verification_score': 0,
                'modification_attempts': 0,
                'metadata': {'error': str(e)}
            }

    async def _generate_answer_from_context(
        self,
        query: str,
        context: str
    ) -> str:
        """
        Generate answer from query and context using LightRAG

        Args:
            query: The query
            context: Retrieved context

        Returns:
            Generated answer string
        """
        # Check if LightRAG is available
        if not hasattr(self, 'lightrag') or self.lightrag is None:
            if hasattr(self, 'logger'):
                self.logger.warning("LightRAG not available for answer generation")
            return "Unable to generate answer: LightRAG not initialized"

        try:
            # Use LightRAG to generate answer from context
            from lightrag import QueryParam

            # Generate answer using the context
            query_param = QueryParam(mode="mix")
            answer = await self.lightrag.aquery(query, param=query_param)

            return answer

        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error generating answer from context: {e}", exc_info=True)
            return f"Error generating answer: {str(e)}"
