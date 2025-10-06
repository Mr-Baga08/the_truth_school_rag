"""
Query Improvement Module for RAG-Anything

This module provides advanced query enhancement capabilities to improve retrieval accuracy
and relevance in RAG systems. It supports multiple improvement strategies including:
- LLM-based intelligent query rewriting
- Rule-based query expansion with domain-specific abbreviations
- Hybrid methods combining LLM and rule-based approaches
- Multi-turn conversation context handling
- Entity extraction from queries

Usage Example:
    ```python
    from raganything.query_improvement import QueryImprover, QueryImprovementConfig

    # Initialize with configuration
    config = QueryImprovementConfig(
        method="hybrid",
        domain="medical",
        expand_abbreviations=True
    )

    improver = QueryImprover(
        llm_func=llm_model_func,
        config=config
    )

    # Improve a query
    result = await improver.improve_query(
        "What is the BP range for HTN patients?",
        domain="medical"
    )

    print(result['improved_query'])
    # Output: "What is the blood pressure range for hypertension patients?"
    ```

Author: RAG-Anything Team
Version: 1.0.0
"""

from __future__ import annotations

import re
import json
import asyncio
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from lightrag.utils import logger


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class QueryImprovementConfig:
    """Configuration for query improvement functionality

    Attributes:
        method: Improvement method - 'llm', 'rules', or 'hybrid'
        domain: Domain context for query improvement (medical, legal, financial, etc.)
        expand_abbreviations: Whether to expand domain-specific abbreviations
        add_domain_keywords: Whether to add relevant domain keywords
        extract_entities: Whether to extract entities from queries
        max_expansion_terms: Maximum number of expansion terms to add
        llm_temperature: Temperature for LLM-based improvement
        llm_max_tokens: Max tokens for LLM response
    """

    method: str = "hybrid"  # 'llm', 'rules', 'hybrid'
    domain: Optional[str] = None  # Domain context
    expand_abbreviations: bool = True
    add_domain_keywords: bool = True
    extract_entities: bool = True
    max_expansion_terms: int = 5
    llm_temperature: float = 0.3
    llm_max_tokens: int = 500
    enable_query_rewriting: bool = True
    preserve_original_intent: bool = True


# =============================================================================
# Domain-Specific Knowledge Bases
# =============================================================================

class DomainKnowledge:
    """Domain-specific abbreviations and keywords"""

    # Medical Domain Abbreviations
    MEDICAL_ABBREVIATIONS = {
        "bp": "blood pressure",
        "hr": "heart rate",
        "htn": "hypertension",
        "dm": "diabetes mellitus",
        "t2dm": "type 2 diabetes mellitus",
        "cad": "coronary artery disease",
        "mi": "myocardial infarction",
        "cvd": "cardiovascular disease",
        "copd": "chronic obstructive pulmonary disease",
        "ckd": "chronic kidney disease",
        "esrd": "end-stage renal disease",
        "aki": "acute kidney injury",
        "chf": "congestive heart failure",
        "af": "atrial fibrillation",
        "dvt": "deep vein thrombosis",
        "pe": "pulmonary embolism",
        "ct": "computed tomography",
        "mri": "magnetic resonance imaging",
        "ecg": "electrocardiogram",
        "ekg": "electrocardiogram",
        "cbc": "complete blood count",
        "bmp": "basic metabolic panel",
        "pt": "prothrombin time",
        "ptt": "partial thromboplastin time",
        "inr": "international normalized ratio",
        "egfr": "estimated glomerular filtration rate",
        "ldl": "low-density lipoprotein",
        "hdl": "high-density lipoprotein",
        "tsh": "thyroid-stimulating hormone",
        "hba1c": "hemoglobin a1c",
        "rbc": "red blood cell",
        "wbc": "white blood cell",
        "ed": "emergency department",
        "icu": "intensive care unit",
        "or": "operating room",
        "iv": "intravenous",
        "po": "per os (by mouth)",
        "prn": "as needed",
        "bid": "twice daily",
        "tid": "three times daily",
        "qid": "four times daily",
        "nsaid": "nonsteroidal anti-inflammatory drug",
        "ace": "angiotensin-converting enzyme",
        "arb": "angiotensin receptor blocker",
        "rx": "prescription treatment",
        "dx": "diagnosis",
        "sx": "symptoms",
        "tx": "treatment therapy",
        "hx": "history",
        "fx": "fracture",
        "px": "physical examination prognosis",
        "sob": "shortness of breath",
        "cp": "chest pain",
        "n/v": "nausea and vomiting",
        "dob": "date of birth",
        "dod": "date of death",
        "stat": "immediately",
        "dnr": "do not resuscitate",
        "ama": "against medical advice",
        "lol": "little old lady",
        "wnl": "within normal limits",
        "nkda": "no known drug allergies",
        "rom": "range of motion",
        "bmi": "body mass index",
        "lvh": "left ventricular hypertrophy",
        "rvh": "right ventricular hypertrophy",
        "sob": "shortness of breath",
        "doe": "dyspnea on exertion",
        "pnd": "paroxysmal nocturnal dyspnea",
        "jvd": "jugular venous distention",
    }

    # Legal Domain Abbreviations
    LEGAL_ABBREVIATIONS = {
        "llc": "limited liability company",
        "inc": "incorporated",
        "corp": "corporation",
        "llp": "limited liability partnership",
        "dba": "doing business as",
        "aka": "also known as",
        "v": "versus",
        "vs": "versus",
        "et al": "and others",
        "ibid": "in the same place",
        "supra": "above",
        "infra": "below",
        "cf": "compare",
        "id": "the same",
        "nda": "non-disclosure agreement",
        "mou": "memorandum of understanding",
        "sla": "service level agreement",
        "ip": "intellectual property",
        "ipr": "intellectual property rights",
        "gdpr": "general data protection regulation",
        "ccpa": "california consumer privacy act",
        "sec": "securities and exchange commission",
        "irs": "internal revenue service",
        "doj": "department of justice",
        "ftc": "federal trade commission",
        "eeo": "equal employment opportunity",
        "eeoc": "equal employment opportunity commission",
        "ada": "americans with disabilities act",
        "fmla": "family and medical leave act",
        "osha": "occupational safety and health administration",
        "nlrb": "national labor relations board",
        "adr": "alternative dispute resolution",
        "arb": "arbitration",
        "mediation": "mediation",
        "plaintiff": "plaintiff",
        "defendant": "defendant",
        "statute": "statute",
        "tort": "tort",
        "negligence": "negligence",
        "breach": "breach",
        "damages": "damages",
        "injunction": "injunction",
        "liability": "liability",
    }

    # Financial Domain Abbreviations
    FINANCIAL_ABBREVIATIONS = {
        "roi": "return on investment",
        "roe": "return on equity",
        "roa": "return on assets",
        "ebitda": "earnings before interest, taxes, depreciation, and amortization",
        "ebit": "earnings before interest and taxes",
        "p/e": "price-to-earnings ratio",
        "eps": "earnings per share",
        "dcf": "discounted cash flow",
        "npv": "net present value",
        "irr": "internal rate of return",
        "wacc": "weighted average cost of capital",
        "capm": "capital asset pricing model",
        "cagr": "compound annual growth rate",
        "yoy": "year over year",
        "qoq": "quarter over quarter",
        "mtd": "month to date",
        "qtd": "quarter to date",
        "ytd": "year to date",
        "fy": "fiscal year",
        "q1": "first quarter",
        "q2": "second quarter",
        "q3": "third quarter",
        "q4": "fourth quarter",
        "gaap": "generally accepted accounting principles",
        "ifrs": "international financial reporting standards",
        "sec": "securities and exchange commission",
        "ipo": "initial public offering",
        "m&a": "mergers and acquisitions",
        "pe": "private equity",
        "vc": "venture capital",
        "etf": "exchange-traded fund",
        "reit": "real estate investment trust",
        "aum": "assets under management",
        "nav": "net asset value",
        "apr": "annual percentage rate",
        "apy": "annual percentage yield",
        "cd": "certificate of deposit",
        "401k": "401k retirement plan",
        "ira": "individual retirement account",
        "hsa": "health savings account",
    }

    # Technical/IT Domain Abbreviations
    TECHNICAL_ABBREVIATIONS = {
        "api": "application programming interface",
        "rest": "representational state transfer",
        "soap": "simple object access protocol",
        "http": "hypertext transfer protocol",
        "https": "hypertext transfer protocol secure",
        "tcp": "transmission control protocol",
        "udp": "user datagram protocol",
        "ip": "internet protocol",
        "dns": "domain name system",
        "url": "uniform resource locator",
        "uri": "uniform resource identifier",
        "html": "hypertext markup language",
        "css": "cascading style sheets",
        "js": "javascript",
        "sql": "structured query language",
        "nosql": "not only sql",
        "crud": "create, read, update, delete",
        "orm": "object-relational mapping",
        "mvc": "model-view-controller",
        "ci/cd": "continuous integration/continuous deployment",
        "devops": "development operations",
        "aws": "amazon web services",
        "gcp": "google cloud platform",
        "azure": "microsoft azure",
        "saas": "software as a service",
        "paas": "platform as a service",
        "iaas": "infrastructure as a service",
        "vm": "virtual machine",
        "docker": "docker container",
        "k8s": "kubernetes",
        "ml": "machine learning",
        "ai": "artificial intelligence",
        "nlp": "natural language processing",
        "cv": "computer vision",
        "dl": "deep learning",
        "nn": "neural network",
        "cnn": "convolutional neural network",
        "rnn": "recurrent neural network",
        "lstm": "long short-term memory",
        "gpt": "generative pre-trained transformer",
        "bert": "bidirectional encoder representations from transformers",
        "rag": "retrieval-augmented generation",
    }

    # Academic/Research Domain Abbreviations
    ACADEMIC_ABBREVIATIONS = {
        "et al": "and others",
        "ibid": "in the same place",
        "op cit": "in the work cited",
        "loc cit": "in the place cited",
        "cf": "compare",
        "viz": "namely",
        "ie": "that is",
        "eg": "for example",
        "nb": "note well",
        "ps": "postscript",
        "qed": "which was to be demonstrated",
        "rsvp": "please respond",
        "phd": "doctor of philosophy",
        "md": "medical doctor",
        "ba": "bachelor of arts",
        "bs": "bachelor of science",
        "ma": "master of arts",
        "ms": "master of science",
        "mba": "master of business administration",
        "jd": "juris doctor",
        "dds": "doctor of dental surgery",
        "dvm": "doctor of veterinary medicine",
        "rct": "randomized controlled trial",
        "anova": "analysis of variance",
        "sd": "standard deviation",
        "sem": "standard error of the mean",
        "ci": "confidence interval",
        "p-value": "probability value",
        "doi": "digital object identifier",
        "isbn": "international standard book number",
        "issn": "international standard serial number",
        "arxiv": "arxiv preprint repository",
        "pubmed": "pubmed database",
        "nih": "national institutes of health",
        "nsf": "national science foundation",
    }

    # Domain Keywords (for context expansion)
    DOMAIN_KEYWORDS = {
        "medical": ["patient", "treatment", "diagnosis", "symptoms", "therapy", "clinical", "medical"],
        "legal": ["law", "legal", "court", "contract", "agreement", "regulation", "compliance"],
        "financial": ["financial", "investment", "revenue", "profit", "market", "trading", "portfolio"],
        "technical": ["technology", "software", "system", "development", "architecture", "infrastructure"],
        "academic": ["research", "study", "analysis", "methodology", "findings", "publication", "peer-reviewed"],
    }

    @classmethod
    def get_abbreviations(cls, domain: Optional[str] = None) -> Dict[str, str]:
        """Get abbreviations for a specific domain or all domains

        Args:
            domain: Domain name (medical, legal, financial, technical, academic) or None for all

        Returns:
            Dictionary of abbreviation mappings
        """
        if domain is None:
            # Combine all domains
            all_abbrevs = {}
            all_abbrevs.update(cls.MEDICAL_ABBREVIATIONS)
            all_abbrevs.update(cls.LEGAL_ABBREVIATIONS)
            all_abbrevs.update(cls.FINANCIAL_ABBREVIATIONS)
            all_abbrevs.update(cls.TECHNICAL_ABBREVIATIONS)
            all_abbrevs.update(cls.ACADEMIC_ABBREVIATIONS)
            return all_abbrevs

        domain_map = {
            "medical": cls.MEDICAL_ABBREVIATIONS,
            "legal": cls.LEGAL_ABBREVIATIONS,
            "financial": cls.FINANCIAL_ABBREVIATIONS,
            "technical": cls.TECHNICAL_ABBREVIATIONS,
            "tech": cls.TECHNICAL_ABBREVIATIONS,
            "it": cls.TECHNICAL_ABBREVIATIONS,
            "academic": cls.ACADEMIC_ABBREVIATIONS,
            "research": cls.ACADEMIC_ABBREVIATIONS,
        }

        return domain_map.get(domain.lower(), {})

    @classmethod
    def get_domain_keywords(cls, domain: Optional[str] = None) -> List[str]:
        """Get relevant keywords for a domain

        Args:
            domain: Domain name or None for all

        Returns:
            List of domain keywords
        """
        if domain is None:
            return []

        return cls.DOMAIN_KEYWORDS.get(domain.lower(), [])


# =============================================================================
# Query Improvement Classes
# =============================================================================

class QueryImprover:
    """Main query improvement class with multiple enhancement strategies

    This class provides comprehensive query improvement capabilities including:
    - LLM-based intelligent query rewriting
    - Rule-based expansion with domain knowledge
    - Hybrid approaches combining multiple methods
    - Entity extraction and query analysis

    Attributes:
        llm_func: LLM function for query improvement
        config: QueryImprovementConfig instance
        domain_knowledge: DomainKnowledge instance
    """

    def __init__(
        self,
        llm_func: Optional[Callable] = None,
        config: Optional[QueryImprovementConfig] = None
    ):
        """Initialize QueryImprover

        Args:
            llm_func: LLM function for query improvement (required for 'llm' or 'hybrid' methods)
            config: Configuration object, if None will use defaults
        """
        self.llm_func = llm_func
        self.config = config or QueryImprovementConfig()
        self.domain_knowledge = DomainKnowledge()

        # Validate configuration
        if self.config.method in ["llm", "hybrid"] and llm_func is None:
            logger.warning(
                f"Query improvement method '{self.config.method}' requires llm_func, "
                "falling back to 'rules' method"
            )
            self.config.method = "rules"

    async def improve_query(
        self,
        query: str,
        domain: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Improve query using configured method

        Args:
            query: Original query string
            domain: Domain context (overrides config domain if provided)
            conversation_history: Previous conversation turns for context

        Returns:
            Dictionary containing:
                - improved_query: Enhanced query string
                - original_query: Original query string
                - method_used: Method used for improvement
                - expansions: List of expansions applied
                - entities: Extracted entities (if enabled)
                - metadata: Additional metadata

        Example:
            ```python
            result = await improver.improve_query(
                "What is BP in HTN patients?",
                domain="medical"
            )
            # result['improved_query'] = "What is blood pressure in hypertension patients?"
            ```
        """
        if not query or not query.strip():
            logger.warning("Empty query provided for improvement")
            return {
                "improved_query": query,
                "original_query": query,
                "method_used": "none",
                "expansions": [],
                "entities": [],
                "metadata": {"error": "Empty query"}
            }

        # Use provided domain or config domain
        effective_domain = domain or self.config.domain

        try:
            # Choose improvement method
            if self.config.method == "llm":
                result = await self._improve_with_llm(query, effective_domain, conversation_history)
            elif self.config.method == "rules":
                result = await self._improve_with_rules(query, effective_domain)
            elif self.config.method == "hybrid":
                result = await self._improve_hybrid(query, effective_domain, conversation_history)
            else:
                logger.warning(f"Unknown improvement method: {self.config.method}")
                result = {
                    "improved_query": query,
                    "original_query": query,
                    "method_used": "none",
                    "expansions": [],
                    "entities": [],
                    "metadata": {"error": f"Unknown method: {self.config.method}"}
                }

            # Extract entities if enabled
            if self.config.extract_entities:
                entities = self._extract_entities(result.get("improved_query", query))
                result["entities"] = entities

            return result

        except Exception as e:
            logger.error(f"Error during query improvement: {e}", exc_info=True)
            return {
                "improved_query": query,  # Return original on error
                "original_query": query,
                "method_used": "error",
                "expansions": [],
                "entities": [],
                "metadata": {"error": str(e)}
            }

    async def _improve_with_llm(
        self,
        query: str,
        domain: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Improve query using LLM-based rewriting

        Args:
            query: Original query
            domain: Domain context
            conversation_history: Conversation history for context

        Returns:
            Improvement result dictionary
        """
        if self.llm_func is None:
            logger.error("LLM function not provided for LLM-based improvement")
            return {
                "improved_query": query,
                "original_query": query,
                "method_used": "error",
                "expansions": [],
                "metadata": {"error": "LLM function not available"}
            }

        try:
            # Build system prompt
            system_prompt = self._build_llm_system_prompt(domain)

            # Build user prompt
            user_prompt = self._build_llm_user_prompt(query, domain, conversation_history)

            # Call LLM
            logger.debug(f"Calling LLM for query improvement: {query[:100]}...")
            response = await self._call_llm_safely(
                prompt=user_prompt,
                system_prompt=system_prompt
            )

            if not response:
                logger.warning("Empty response from LLM, using original query")
                return {
                    "improved_query": query,
                    "original_query": query,
                    "method_used": "llm_failed",
                    "expansions": [],
                    "metadata": {"error": "Empty LLM response"}
                }

            # Parse LLM response
            parsed = self._parse_llm_response(response)

            return {
                "improved_query": parsed.get("improved_query", query),
                "original_query": query,
                "method_used": "llm",
                "expansions": parsed.get("expansions", []),
                "metadata": {
                    "reasoning": parsed.get("reasoning", ""),
                    "domain": domain,
                    "llm_response": response
                }
            }

        except Exception as e:
            logger.error(f"Error in LLM-based improvement: {e}", exc_info=True)
            return {
                "improved_query": query,
                "original_query": query,
                "method_used": "llm_error",
                "expansions": [],
                "metadata": {"error": str(e)}
            }

    async def _improve_with_rules(
        self,
        query: str,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """Improve query using rule-based expansion

        Args:
            query: Original query
            domain: Domain context

        Returns:
            Improvement result dictionary
        """
        improved_query = query
        expansions = []

        try:
            # Step 1: Expand abbreviations
            if self.config.expand_abbreviations:
                improved_query, abbrev_expansions = self._expand_abbreviations(
                    improved_query, domain
                )
                expansions.extend(abbrev_expansions)

            # Step 2: Add domain keywords
            if self.config.add_domain_keywords and domain:
                improved_query, keyword_additions = self._add_domain_keywords(
                    improved_query, domain
                )
                expansions.extend(keyword_additions)

            logger.debug(f"Rule-based improvement: '{query}' -> '{improved_query}'")

            return {
                "improved_query": improved_query,
                "original_query": query,
                "method_used": "rules",
                "expansions": expansions,
                "metadata": {
                    "domain": domain,
                    "abbreviations_expanded": len([e for e in expansions if e["type"] == "abbreviation"]),
                    "keywords_added": len([e for e in expansions if e["type"] == "keyword"])
                }
            }

        except Exception as e:
            logger.error(f"Error in rule-based improvement: {e}", exc_info=True)
            return {
                "improved_query": query,
                "original_query": query,
                "method_used": "rules_error",
                "expansions": [],
                "metadata": {"error": str(e)}
            }

    async def _improve_hybrid(
        self,
        query: str,
        domain: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Improve query using hybrid approach (rules + LLM)

        Args:
            query: Original query
            domain: Domain context
            conversation_history: Conversation history

        Returns:
            Improvement result dictionary
        """
        try:
            # Step 1: Apply rule-based improvements first
            rules_result = await self._improve_with_rules(query, domain)
            intermediate_query = rules_result["improved_query"]

            # Step 2: Apply LLM improvements to the rule-improved query
            llm_result = await self._improve_with_llm(
                intermediate_query, domain, conversation_history
            )

            # Combine results
            all_expansions = rules_result["expansions"] + llm_result.get("expansions", [])

            return {
                "improved_query": llm_result["improved_query"],
                "original_query": query,
                "method_used": "hybrid",
                "expansions": all_expansions,
                "metadata": {
                    "domain": domain,
                    "rules_result": rules_result["improved_query"],
                    "llm_result": llm_result["improved_query"],
                    "total_expansions": len(all_expansions)
                }
            }

        except Exception as e:
            logger.error(f"Error in hybrid improvement: {e}", exc_info=True)
            # Fall back to rules-only on error
            return await self._improve_with_rules(query, domain)

    def _expand_abbreviations(
        self,
        query: str,
        domain: Optional[str] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Expand abbreviations in query

        Args:
            query: Query string
            domain: Domain for abbreviation lookup

        Returns:
            (expanded_query, expansions_list)
        """
        abbreviations = self.domain_knowledge.get_abbreviations(domain)
        expanded_query = query
        expansions = []

        # Sort abbreviations by length (longest first) to handle overlaps
        sorted_abbrevs = sorted(abbreviations.items(), key=lambda x: len(x[0]), reverse=True)

        for abbrev, expansion in sorted_abbrevs:
            # Create pattern for whole-word matching (case-insensitive)
            pattern = r'\b' + re.escape(abbrev) + r'\b'

            # Check if abbreviation exists in query
            matches = list(re.finditer(pattern, expanded_query, re.IGNORECASE))

            for match in matches:
                original = match.group()
                # Preserve case of first letter
                if original[0].isupper():
                    replacement = expansion.capitalize()
                else:
                    replacement = expansion.lower()

                # Replace in query
                expanded_query = expanded_query[:match.start()] + replacement + expanded_query[match.end():]

                # Record expansion
                expansions.append({
                    "type": "abbreviation",
                    "original": original,
                    "expanded": replacement,
                    "domain": domain
                })

                logger.debug(f"Expanded abbreviation: '{original}' -> '{replacement}'")

                # Re-create pattern after replacement
                break

        return expanded_query, expansions

    def _add_domain_keywords(
        self,
        query: str,
        domain: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Add relevant domain keywords to query

        Args:
            query: Query string
            domain: Domain for keyword lookup

        Returns:
            (enhanced_query, additions_list)
        """
        keywords = self.domain_knowledge.get_domain_keywords(domain)
        additions = []

        if not keywords:
            return query, additions

        # Check which keywords are already present
        query_lower = query.lower()
        missing_keywords = [kw for kw in keywords if kw not in query_lower]

        # Limit additions
        keywords_to_add = missing_keywords[:self.config.max_expansion_terms]

        if keywords_to_add:
            # Add keywords as context
            keyword_context = " ".join(keywords_to_add)
            enhanced_query = f"{query} ({keyword_context})"

            for kw in keywords_to_add:
                additions.append({
                    "type": "keyword",
                    "keyword": kw,
                    "domain": domain
                })

            logger.debug(f"Added {len(keywords_to_add)} domain keywords to query")

            return enhanced_query, additions

        return query, additions

    def _extract_entities(self, query: str) -> List[Dict[str, Any]]:
        """Extract entities from query using simple heuristics

        Args:
            query: Query string

        Returns:
            List of extracted entities
        """
        entities = []

        # Extract capitalized words (potential named entities)
        capitalized_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        capitalized_matches = re.findall(capitalized_pattern, query)

        for match in capitalized_matches:
            entities.append({
                "text": match,
                "type": "named_entity",
                "method": "capitalization"
            })

        # Extract quoted phrases
        quoted_pattern = r'"([^"]+)"'
        quoted_matches = re.findall(quoted_pattern, query)

        for match in quoted_matches:
            entities.append({
                "text": match,
                "type": "quoted_phrase",
                "method": "quotes"
            })

        # Extract numbers with units
        number_pattern = r'\b\d+(?:\.\d+)?\s*(?:mg|ml|kg|lb|cm|mm|km|mph|%|dollars?|euros?)\b'
        number_matches = re.findall(number_pattern, query, re.IGNORECASE)

        for match in number_matches:
            entities.append({
                "text": match,
                "type": "measurement",
                "method": "pattern"
            })

        return entities

    def _build_llm_system_prompt(self, domain: Optional[str] = None) -> str:
        """Build system prompt for LLM-based improvement

        Args:
            domain: Domain context

        Returns:
            System prompt string
        """
        base_prompt = """You are an expert query optimization assistant for RAG (Retrieval-Augmented Generation) systems.

Your task is to improve user queries to maximize retrieval accuracy and relevance. You should:
1. Clarify ambiguous terms
2. Expand abbreviations and acronyms
3. Add relevant context
4. Rephrase for better semantic matching
5. Preserve the original intent and meaning

Return your response in JSON format with the following structure:
{
    "improved_query": "the improved query string",
    "reasoning": "brief explanation of changes made",
    "expansions": ["list of specific improvements applied"]
}"""

        if domain:
            domain_context = f"\n\nDomain Context: This query is from the {domain} domain. Apply domain-specific knowledge and terminology when improving the query."
            base_prompt += domain_context

        return base_prompt

    def _build_llm_user_prompt(
        self,
        query: str,
        domain: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Build user prompt for LLM

        Args:
            query: Original query
            domain: Domain context
            conversation_history: Conversation history

        Returns:
            User prompt string
        """
        prompt_parts = [f"Original Query: {query}"]

        if domain:
            prompt_parts.append(f"Domain: {domain}")

        if conversation_history:
            history_str = "\n".join([
                f"{turn['role']}: {turn['content']}"
                for turn in conversation_history[-3:]  # Last 3 turns
            ])
            prompt_parts.append(f"Conversation History:\n{history_str}")

        prompt_parts.append("\nPlease improve this query and return the response in the specified JSON format.")

        return "\n\n".join(prompt_parts)

    async def _call_llm_safely(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_retries: int = 2
    ) -> str:
        """Call LLM with error handling and retry logic

        Args:
            prompt: User prompt
            system_prompt: System prompt
            max_retries: Maximum number of retry attempts

        Returns:
            LLM response string
        """
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(self.llm_func):
                    response = await self.llm_func(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=self.config.llm_temperature,
                        max_tokens=self.config.llm_max_tokens
                    )
                else:
                    response = self.llm_func(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=self.config.llm_temperature,
                        max_tokens=self.config.llm_max_tokens
                    )

                # Check if response is empty or only whitespace
                if response and response.strip():
                    return response
                else:
                    logger.warning(f"Empty LLM response on attempt {attempt + 1}/{max_retries + 1}")
                    if attempt < max_retries:
                        await asyncio.sleep(0.5)  # Small delay before retry
                        continue
                    return ""  # Return empty string after all retries

            except Exception as e:
                last_error = e
                logger.error(f"Error calling LLM (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt < max_retries:
                    await asyncio.sleep(0.5)
                    continue
                raise

        # If we get here, all retries failed
        if last_error:
            raise last_error
        return ""

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM JSON response with robust error handling

        Args:
            response: LLM response string

        Returns:
            Parsed dictionary
        """
        try:
            # Clean response - remove markdown code blocks if present
            cleaned_response = self._clean_json_response(response)

            # Parse JSON
            parsed = json.loads(cleaned_response)

            # Validate required fields
            if "improved_query" not in parsed:
                logger.warning("LLM response missing 'improved_query' field")
                parsed["improved_query"] = response  # Use raw response as fallback

            return parsed

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            # Return response as improved query
            return {
                "improved_query": response.strip(),
                "reasoning": "Failed to parse JSON response",
                "expansions": []
            }

    def _clean_json_response(self, response: str) -> str:
        """Clean JSON response by removing markdown code blocks and extra whitespace

        Args:
            response: Raw LLM response

        Returns:
            Cleaned JSON string
        """
        # Remove markdown code blocks
        cleaned = re.sub(r'```json\s*', '', response)
        cleaned = re.sub(r'```\s*', '', cleaned)

        # Strip whitespace
        cleaned = cleaned.strip()

        return cleaned


# =============================================================================
# Query Rewriter for Conversational Context
# =============================================================================

class QueryRewriter:
    """Query rewriter for multi-turn conversations

    This class handles query rewriting in conversational contexts where
    queries may reference previous turns and need contextualization.

    Example:
        ```python
        rewriter = QueryRewriter(llm_func=llm_model_func)

        # First query
        result1 = await rewriter.rewrite_query(
            "What is hypertension?",
            conversation_history=[]
        )

        # Follow-up query with context
        result2 = await rewriter.rewrite_query(
            "What are its symptoms?",
            conversation_history=[
                {"role": "user", "content": "What is hypertension?"},
                {"role": "assistant", "content": "Hypertension is high blood pressure..."}
            ]
        )
        # result2['rewritten_query'] might be: "What are the symptoms of hypertension?"
        ```
    """

    def __init__(self, llm_func: Optional[Callable] = None, config: Optional[QueryImprovementConfig] = None):
        """Initialize QueryRewriter

        Args:
            llm_func: LLM function for query rewriting
            config: Configuration object
        """
        self.llm_func = llm_func
        self.config = config or QueryImprovementConfig()

    async def rewrite_query(
        self,
        query: str,
        conversation_history: List[Dict[str, str]],
        max_history_turns: int = 5
    ) -> Dict[str, Any]:
        """Rewrite query based on conversation history

        Args:
            query: Current query
            conversation_history: List of previous conversation turns
            max_history_turns: Maximum history turns to consider

        Returns:
            Dictionary with rewritten_query and metadata
        """
        if not conversation_history:
            return {
                "rewritten_query": query,
                "original_query": query,
                "context_used": False
            }

        if self.llm_func is None:
            logger.warning("LLM function not provided for query rewriting")
            return {
                "rewritten_query": query,
                "original_query": query,
                "context_used": False,
                "error": "LLM function not available"
            }

        try:
            # Check if query needs rewriting (has pronouns or references)
            if not self._needs_rewriting(query):
                return {
                    "rewritten_query": query,
                    "original_query": query,
                    "context_used": False,
                    "reason": "No pronouns or references detected"
                }

            # Build rewriting prompt
            system_prompt = """You are a query rewriting assistant. Your task is to rewrite user queries to be self-contained by incorporating relevant context from the conversation history.

Rules:
1. Replace pronouns (it, they, this, that) with their specific referents
2. Add necessary context to make the query standalone
3. Preserve the original question's intent
4. Keep the rewritten query concise and natural

Return only the rewritten query, nothing else."""

            # Format conversation history
            recent_history = conversation_history[-max_history_turns:]
            history_str = "\n".join([
                f"{turn['role'].upper()}: {turn['content']}"
                for turn in recent_history
            ])

            user_prompt = f"""Conversation History:
{history_str}

Current Query: {query}

Rewritten Query:"""

            # Call LLM
            if asyncio.iscoroutinefunction(self.llm_func):
                rewritten = await self.llm_func(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.llm_max_tokens
                )
            else:
                rewritten = self.llm_func(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.llm_max_tokens
                )

            rewritten = rewritten.strip()

            logger.debug(f"Rewrote query: '{query}' -> '{rewritten}'")

            return {
                "rewritten_query": rewritten,
                "original_query": query,
                "context_used": True,
                "history_turns_used": len(recent_history)
            }

        except Exception as e:
            logger.error(f"Error in query rewriting: {e}", exc_info=True)
            return {
                "rewritten_query": query,
                "original_query": query,
                "context_used": False,
                "error": str(e)}

    def _needs_rewriting(self, query: str) -> bool:
        """Check if query contains pronouns or references that need rewriting

        Args:
            query: Query string

        Returns:
            True if rewriting is needed
        """
        # Common pronouns and references
        pronouns = r'\b(it|they|them|this|that|these|those|he|she|his|her|their)\b'

        # Check for pronouns
        if re.search(pronouns, query, re.IGNORECASE):
            return True

        # Check for incomplete questions
        incomplete_patterns = [
            r'^\s*(what|how|why|when|where)\s+about',  # "What about..."
            r'^\s*(and|or|but)\s+',  # Starting with conjunction
        ]

        for pattern in incomplete_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True

        return False


# =============================================================================
# Mixin Class for Integration
# =============================================================================

class QueryImprovementMixin:
    """
    Mixin providing query improvement functionality to RAGAnything

    This mixin adds query improvement capabilities to the RAGAnything class.
    It provides methods to enhance queries before retrieval using:
    - LLM-based query rewriting
    - Rule-based expansion with domain abbreviations
    - Hybrid approaches combining multiple techniques
    - Conversation context handling

    The mixin expects the following attributes to be present:
    - self.query_improver: QueryImprover instance (optional)
    - self.query_rewriter: QueryRewriter instance (optional)
    - self.config: RAGAnythingConfig instance
    - self.logger: Logger instance
    """

    async def _apply_query_improvement(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Apply query improvement to input query

        This method enhances the user's query using the configured improvement
        method (llm, rules, or hybrid). It can also incorporate conversation
        context for better understanding.

        Args:
            query: Original user query
            conversation_history: Optional conversation history for context

        Returns:
            Dict containing:
                - improved_query: The enhanced query string
                - original_query: The original query string
                - method_used: Which method was used (llm/rules/hybrid)
                - expansions: List of expansions applied (if any)
                - entities: Extracted entities (if enabled)
                - metadata: Additional improvement metadata

        Example:
            result = await self._apply_query_improvement(
                "What is HTN treatment?",
                conversation_history=[
                    {"role": "user", "content": "Tell me about cardiovascular diseases"},
                    {"role": "assistant", "content": "Cardiovascular diseases..."}
                ]
            )
            # result['improved_query'] might be:
            # "What is hypertension (HTN) treatment in the context of cardiovascular diseases?"
        """
        # Check if query improver is available
        if not hasattr(self, 'query_improver') or self.query_improver is None:
            if hasattr(self, 'logger'):
                self.logger.debug(
                    "Query improver not initialized, returning original query"
                )
            return {
                'improved_query': query,
                'original_query': query,
                'method_used': 'none',
                'expansions': [],
                'entities': [],
                'metadata': {'reason': 'query_improver_not_initialized'}
            }

        try:
            # Get domain from config if available
            domain = getattr(self.config, 'domain', None) if hasattr(self, 'config') else None

            if hasattr(self, 'logger'):
                self.logger.debug(
                    f"Applying query improvement (method: {self.query_improver.config.method}, "
                    f"domain: {domain})"
                )

            # Apply query improvement
            improvement_result = await self.query_improver.improve_query(
                query=query,
                domain=domain,
                conversation_history=conversation_history
            )

            if hasattr(self, 'logger'):
                self.logger.debug(
                    f"Query improvement complete: "
                    f"'{improvement_result['original_query'][:50]}...' -> "
                    f"'{improvement_result['improved_query'][:50]}...'"
                )

            return improvement_result

        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error applying query improvement: {e}", exc_info=True)

            # Return original query on error
            return {
                'improved_query': query,
                'original_query': query,
                'method_used': 'none',
                'expansions': [],
                'entities': [],
                'metadata': {'error': str(e)}
            }

    async def _rewrite_query_with_context(
        self,
        query: str,
        conversation_history: List[Dict[str, str]],
        max_history_turns: int = 5
    ) -> Dict[str, Any]:
        """
        Rewrite query using conversation context

        This method rewrites the query to incorporate conversation context,
        resolving pronouns and references to previous messages.

        Args:
            query: Original user query (may contain pronouns/references)
            conversation_history: Previous conversation turns
            max_history_turns: Maximum number of history turns to consider

        Returns:
            Dict containing:
                - rewritten_query: The context-aware rewritten query
                - original_query: The original query string
                - context_used: List of context snippets used
                - metadata: Rewriting metadata

        Example:
            # After discussing "machine learning"
            result = await self._rewrite_query_with_context(
                "How does it work?",
                conversation_history=[
                    {"role": "user", "content": "What is machine learning?"},
                    {"role": "assistant", "content": "Machine learning is..."}
                ]
            )
            # result['rewritten_query'] might be:
            # "How does machine learning work?"
        """
        # Check if query rewriter is available
        if not hasattr(self, 'query_rewriter') or self.query_rewriter is None:
            if hasattr(self, 'logger'):
                self.logger.debug(
                    "Query rewriter not initialized, returning original query"
                )
            return {
                'rewritten_query': query,
                'original_query': query,
                'context_used': [],
                'metadata': {'reason': 'query_rewriter_not_initialized'}
            }

        # If no conversation history, return original
        if not conversation_history:
            if hasattr(self, 'logger'):
                self.logger.debug(
                    "No conversation history provided, skipping rewrite"
                )
            return {
                'rewritten_query': query,
                'original_query': query,
                'context_used': [],
                'metadata': {'reason': 'no_conversation_history'}
            }

        try:
            if hasattr(self, 'logger'):
                self.logger.debug(
                    f"Rewriting query with {len(conversation_history)} history turns"
                )

            # Apply query rewriting
            rewrite_result = await self.query_rewriter.rewrite_query(
                query=query,
                conversation_history=conversation_history,
                max_history_turns=max_history_turns
            )

            if hasattr(self, 'logger'):
                self.logger.debug(
                    f"Query rewrite complete: "
                    f"'{rewrite_result['original_query'][:50]}...' -> "
                    f"'{rewrite_result['rewritten_query'][:50]}...'"
                )

            return rewrite_result

        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error rewriting query: {e}", exc_info=True)

            # Return original query on error
            return {
                'rewritten_query': query,
                'original_query': query,
                'context_used': [],
                'metadata': {'error': str(e)}
            }
