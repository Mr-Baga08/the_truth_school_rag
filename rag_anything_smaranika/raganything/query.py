# """
# Query functionality for RAGAnything

# Contains all query-related methods for both text and multimodal queries
# """

# import json
# import hashlib
# import re
# from typing import Dict, List, Any
# from pathlib import Path
# from lightrag import QueryParam
# from lightrag.utils import always_get_an_event_loop
# from raganything.prompt import PROMPTS
# from raganything.utils import (
#     get_processor_for_type,
#     encode_image_to_base64,
#     validate_image_file,
# )
# # Add these imports
# from raganything.query_improvement import QueryImprovementMixin
# from raganything.verification import DualLLMVerificationMixin


# class QueryMixin(QueryImprovementMixin, DualLLMVerificationMixin):
#     """QueryMixin class containing query functionality for RAGAnything"""

#     def _generate_multimodal_cache_key(
#         self, query: str, multimodal_content: List[Dict[str, Any]], mode: str, **kwargs
#     ) -> str:
#         """
#         Generate cache key for multimodal query

#         Args:
#             query: Base query text
#             multimodal_content: List of multimodal content
#             mode: Query mode
#             **kwargs: Additional parameters

#         Returns:
#             str: Cache key hash
#         """
#         # Create a normalized representation of the query parameters
#         cache_data = {
#             "query": query.strip(),
#             "mode": mode,
#         }

#         # Normalize multimodal content for stable caching
#         normalized_content = []
#         if multimodal_content:
#             for item in multimodal_content:
#                 if isinstance(item, dict):
#                     normalized_item = {}
#                     for key, value in item.items():
#                         # For file paths, use basename to make cache more portable
#                         if key in [
#                             "img_path",
#                             "image_path",
#                             "file_path",
#                         ] and isinstance(value, str):
#                             normalized_item[key] = Path(value).name
#                         # For large content, create a hash instead of storing directly
#                         elif (
#                             key in ["table_data", "table_body"]
#                             and isinstance(value, str)
#                             and len(value) > 200
#                         ):
#                             normalized_item[f"{key}_hash"] = hashlib.md5(
#                                 value.encode()
#                             ).hexdigest()
#                         else:
#                             normalized_item[key] = value
#                     normalized_content.append(normalized_item)
#                 else:
#                     normalized_content.append(item)

#         cache_data["multimodal_content"] = normalized_content

#         # Add relevant kwargs to cache data
#         relevant_kwargs = {
#             k: v
#             for k, v in kwargs.items()
#             if k
#             in [
#                 "stream",
#                 "response_type",
#                 "top_k",
#                 "max_tokens",
#                 "temperature",
#                 # "only_need_context",
#                 # "only_need_prompt",
#             ]
#         }
#         cache_data.update(relevant_kwargs)

#         # Generate hash from the cache data
#         cache_str = json.dumps(cache_data, sort_keys=True, ensure_ascii=False)
#         cache_hash = hashlib.md5(cache_str.encode()).hexdigest()

#         return f"multimodal_query:{cache_hash}"

#     # async def aquery(self, query: str, mode: str = "mix", **kwargs) -> str:
#     #     """
#     #     Pure text query - directly calls LightRAG's query functionality

#     #     Args:
#     #         query: Query text
#     #         mode: Query mode ("local", "global", "hybrid", "naive", "mix", "bypass")
#     #         **kwargs: Other query parameters, will be passed to QueryParam
#     #             - vlm_enhanced: bool, default True when vision_model_func is available.
#     #               If True, will parse image paths in retrieved context and replace them
#     #               with base64 encoded images for VLM processing.

#     #     Returns:
#     #         str: Query result
#     #     """
#     #     if self.lightrag is None:
#     #         raise ValueError(
#     #             "No LightRAG instance available. Please process documents first or provide a pre-initialized LightRAG instance."
#     #         )

#     #     # Check if VLM enhanced query should be used
#     #     vlm_enhanced = kwargs.pop("vlm_enhanced", None)

#     #     # Auto-determine VLM enhanced based on availability
#     #     if vlm_enhanced is None:
#     #         vlm_enhanced = (
#     #             hasattr(self, "vision_model_func")
#     #             and self.vision_model_func is not None
#     #         )

#     #     # Use VLM enhanced query if enabled and available
#     #     if (
#     #         vlm_enhanced
#     #         and hasattr(self, "vision_model_func")
#     #         and self.vision_model_func
#     #     ):
#     #         return await self.aquery_vlm_enhanced(query, mode=mode, **kwargs)
#     #     elif vlm_enhanced and (
#     #         not hasattr(self, "vision_model_func") or not self.vision_model_func
#     #     ):
#     #         self.logger.warning(
#     #             "VLM enhanced query requested but vision_model_func is not available, falling back to normal query"
#     #         )

#     #     # Create query parameters
#     #     query_param = QueryParam(mode=mode, **kwargs)

#     #     self.logger.info(f"Executing text query: {query[:100]}...")
#     #     self.logger.info(f"Query mode: {mode}")

#     #     # Call LightRAG's query method
#     #     result = await self.lightrag.aquery(query, param=query_param)

#     #     self.logger.info("Text query completed")
#     #     return result

#     # async def aquery_with_multimodal(
#     #     self,
#     #     query: str,
#     #     multimodal_content: List[Dict[str, Any]] = None,
#     #     mode: str = "mix",
#     #     **kwargs,
#     # ) -> str:
#     #     """
#     #     Multimodal query - combines text and multimodal content for querying

#     #     Args:
#     #         query: Base query text
#     #         multimodal_content: List of multimodal content, each element contains:
#     #             - type: Content type ("image", "table", "equation", etc.)
#     #             - Other fields depend on type (e.g., img_path, table_data, latex, etc.)
#     #         mode: Query mode ("local", "global", "hybrid", "naive", "mix", "bypass")
#     #         **kwargs: Other query parameters, will be passed to QueryParam

#     #     Returns:
#     #         str: Query result

#     #     Examples:
#     #         # Pure text query
#     #         result = await rag.query_with_multimodal("What is machine learning?")

#     #         # Image query
#     #         result = await rag.query_with_multimodal(
#     #             "Analyze the content in this image",
#     #             multimodal_content=[{
#     #                 "type": "image",
#     #                 "img_path": "./image.jpg"
#     #             }]
#     #         )

#     #         # Table query
#     #         result = await rag.query_with_multimodal(
#     #             "Analyze the data trends in this table",
#     #             multimodal_content=[{
#     #                 "type": "table",
#     #                 "table_data": "Name,Age\nAlice,25\nBob,30"
#     #             }]
#     #         )
#     #     """
#     #     # Ensure LightRAG is initialized
#     #     await self._ensure_lightrag_initialized()

#     #     self.logger.info(f"Executing multimodal query: {query[:100]}...")
#     #     self.logger.info(f"Query mode: {mode}")

#     #     # If no multimodal content, fallback to pure text query
#     #     if not multimodal_content:
#     #         self.logger.info("No multimodal content provided, executing text query")
#     #         return await self.aquery(query, mode=mode, **kwargs)

#     #     # Generate cache key for multimodal query
#     #     cache_key = self._generate_multimodal_cache_key(
#     #         query, multimodal_content, mode, **kwargs
#     #     )

#     #     # Check cache if available and enabled
#     #     cached_result = None
#     #     if (
#     #         hasattr(self, "lightrag")
#     #         and self.lightrag
#     #         and hasattr(self.lightrag, "llm_response_cache")
#     #         and self.lightrag.llm_response_cache
#     #     ):
#     #         if self.lightrag.llm_response_cache.global_config.get(
#     #             "enable_llm_cache", True
#     #         ):
#     #             try:
#     #                 cached_result = await self.lightrag.llm_response_cache.get_by_id(
#     #                     cache_key
#     #                 )
#     #                 if cached_result and isinstance(cached_result, dict):
#     #                     result_content = cached_result.get("return")
#     #                     if result_content:
#     #                         self.logger.info(
#     #                             f"Multimodal query cache hit: {cache_key[:16]}..."
#     #                         )
#     #                         return result_content
#     #             except Exception as e:
#     #                 self.logger.debug(f"Error accessing multimodal query cache: {e}")

#     #     # Process multimodal content to generate enhanced query text
#     #     enhanced_query = await self._process_multimodal_query_content(
#     #         query, multimodal_content
#     #     )

#     #     self.logger.info(
#     #         f"Generated enhanced query length: {len(enhanced_query)} characters"
#     #     )

#     #     # Execute enhanced query
#     #     result = await self.aquery(enhanced_query, mode=mode, **kwargs)

#     #     # Save to cache if available and enabled
#     #     if (
#     #         hasattr(self, "lightrag")
#     #         and self.lightrag
#     #         and hasattr(self.lightrag, "llm_response_cache")
#     #         and self.lightrag.llm_response_cache
#     #     ):
#     #         if self.lightrag.llm_response_cache.global_config.get(
#     #             "enable_llm_cache", True
#     #         ):
#     #             try:
#     #                 # Create cache entry for multimodal query
#     #                 cache_entry = {
#     #                     "return": result,
#     #                     "cache_type": "multimodal_query",
#     #                     "original_query": query,
#     #                     "multimodal_content_count": len(multimodal_content),
#     #                     "mode": mode,
#     #                 }

#     #                 await self.lightrag.llm_response_cache.upsert(
#     #                     {cache_key: cache_entry}
#     #                 )
#     #                 self.logger.info(
#     #                     f"Saved multimodal query result to cache: {cache_key[:16]}..."
#     #                 )
#     #             except Exception as e:
#     #                 self.logger.debug(f"Error saving multimodal query to cache: {e}")

#     #     # Ensure cache is persisted to disk
#     #     if (
#     #         hasattr(self, "lightrag")
#     #         and self.lightrag
#     #         and hasattr(self.lightrag, "llm_response_cache")
#     #         and self.lightrag.llm_response_cache
#     #     ):
#     #         try:
#     #             await self.lightrag.llm_response_cache.index_done_callback()
#     #         except Exception as e:
#     #             self.logger.debug(f"Error persisting multimodal query cache: {e}")

#     #     self.logger.info("Multimodal query completed")
#     #     return result

#     async def aquery(self, query: str, mode: str = "mix", **kwargs) -> str:
#         """
#         Pure text query with optional query improvement and verification
        
#         Args:
#             query: Query text
#             mode: Query mode ("local", "global", "hybrid", "naive", "mix", "bypass")
#             **kwargs: Other query parameters
#                 - enable_query_improvement: bool, override config setting
#                 - enable_verification: bool, override config setting
#                 - return_verification_info: bool, return detailed verification info
        
#         Returns:
#             str: Query result (or dict if return_verification_info=True)
#         """
#         if self.lightrag is None:
#             raise ValueError(
#                 "No LightRAG instance available. Please process documents first or provide a pre-initialized LightRAG instance."
#             )
        
#         # Check override flags
#         use_query_improvement = kwargs.pop('enable_query_improvement', 
#                                            getattr(self.config, 'enable_query_improvement', False))
#         use_verification = kwargs.pop('enable_verification', 
#                                        getattr(self.config, 'enable_dual_llm_verification', False))
#         return_verification_info = kwargs.pop('return_verification_info', False)
        
#         original_query = query
#         query_improvement_result = None
        
#         # Step 1: Apply query improvement if enabled
#         if use_query_improvement and hasattr(self, 'query_improver') and self.query_improver:
#             self.logger.info("Applying query improvement...")
#             query_improvement_result = await self._apply_query_improvement(query)
#             query = query_improvement_result["improved_query"]
#             self.logger.info(f"Query improved: '{original_query[:50]}...' -> '{query[:50]}...'")
        
#         # Step 2: Check VLM enhanced query
#         vlm_enhanced = kwargs.pop("vlm_enhanced", None)
#         if vlm_enhanced is None:
#             vlm_enhanced = (
#                 hasattr(self, "vision_model_func") and self.vision_model_func is not None
#             )
        
#         # If using VLM enhanced or verification is disabled, use existing flow
#         if vlm_enhanced or not use_verification:
#             if vlm_enhanced and hasattr(self, "vision_model_func") and self.vision_model_func:
#                 result = await self.aquery_vlm_enhanced(query, mode=mode, **kwargs)
#             else:
#                 from lightrag import QueryParam
#                 query_param = QueryParam(mode=mode, **kwargs)
#                 result = await self.lightrag.aquery(query, param=query_param)
            
#             if return_verification_info:
#                 return {
#                     "answer": result,
#                     "original_query": original_query,
#                     "improved_query": query if query_improvement_result else original_query,
#                     "query_improvement": query_improvement_result,
#                     "verification_passed": True,
#                     "verification_score": 10.0
#                 }
#             return result
        
#         # Step 3: Generate with verification
#         if use_verification and hasattr(self, 'answer_verifier') and self.answer_verifier:
#             self.logger.info("Using dual-LLM verification...")
            
#             # Get context without final answer
#             from lightrag import QueryParam
#             query_param = QueryParam(mode=mode, only_need_context=True, **kwargs)
#             context = await self.lightrag.aquery(query, param=query_param)
            
#             # Generate with verification
#             verification_result = await self._generate_with_verification(
#                 query=query,
#                 context=context,
#                 original_query=original_query
#             )
            
#             if return_verification_info:
#                 return {
#                     "answer": verification_result["answer"],
#                     "original_query": original_query,
#                     "improved_query": query if query_improvement_result else original_query,
#                     "query_improvement": query_improvement_result,
#                     "verification_passed": verification_result["verification_passed"],
#                     "verification_score": verification_result["verification_score"],
#                     "modification_attempts": verification_result["modification_attempts"],
#                     "verification_history": verification_result.get("verification_history", [])
#                 }
            
#             return verification_result["answer"]
        
#         # Fallback to normal query
#         from lightrag import QueryParam
#         query_param = QueryParam(mode=mode, **kwargs)
#         result = await self.lightrag.aquery(query, param=query_param)
        
#         if return_verification_info:
#             return {
#                 "answer": result,
#                 "original_query": original_query,
#                 "improved_query": query if query_improvement_result else original_query,
#                 "query_improvement": query_improvement_result
#             }
        
#         return result

#     async def aquery_vlm_enhanced(self, query: str, mode: str = "mix", **kwargs) -> str:
#         """
#         VLM enhanced query - replaces image paths in retrieved context with base64 encoded images for VLM processing

#         Args:
#             query: User query
#             mode: Underlying LightRAG query mode
#             **kwargs: Other query parameters

#         Returns:
#             str: VLM query result
#         """
#         # Ensure VLM is available
#         if not hasattr(self, "vision_model_func") or not self.vision_model_func:
#             raise ValueError(
#                 "VLM enhanced query requires vision_model_func. "
#                 "Please provide a vision model function when initializing RAGAnything."
#             )

#         # Ensure LightRAG is initialized
#         await self._ensure_lightrag_initialized()

#         self.logger.info(f"Executing VLM enhanced query: {query[:100]}...")

#         # Clear previous image cache
#         if hasattr(self, "_current_images_base64"):
#             delattr(self, "_current_images_base64")

#         # 1. Get original retrieval prompt (without generating final answer)
#         query_param = QueryParam(mode=mode, only_need_prompt=True, **kwargs)
#         raw_prompt = await self.lightrag.aquery(query, param=query_param)

#         self.logger.debug("Retrieved raw prompt from LightRAG")

#         # 2. Extract and process image paths
#         enhanced_prompt, images_found = await self._process_image_paths_for_vlm(
#             raw_prompt
#         )

#         if not images_found:
#             self.logger.info("No valid images found, falling back to normal query")
#             # Fallback to normal query
#             query_param = QueryParam(mode=mode, **kwargs)
#             return await self.lightrag.aquery(query, param=query_param)

#         self.logger.info(f"Processed {images_found} images for VLM")

#         # 3. Build VLM message format
#         messages = self._build_vlm_messages_with_images(enhanced_prompt, query)

#         # 4. Call VLM for question answering
#         result = await self._call_vlm_with_multimodal_content(messages)

#         self.logger.info("VLM enhanced query completed")
#         return result

#     async def _process_multimodal_query_content(
#         self, base_query: str, multimodal_content: List[Dict[str, Any]]
#     ) -> str:
#         """
#         Process multimodal query content to generate enhanced query text

#         Args:
#             base_query: Base query text
#             multimodal_content: List of multimodal content

#         Returns:
#             str: Enhanced query text
#         """
#         self.logger.info("Starting multimodal query content processing...")

#         enhanced_parts = [f"User query: {base_query}"]

#         for i, content in enumerate(multimodal_content):
#             content_type = content.get("type", "unknown")
#             self.logger.info(
#                 f"Processing {i+1}/{len(multimodal_content)} multimodal content: {content_type}"
#             )

#             try:
#                 # Get appropriate processor
#                 processor = get_processor_for_type(self.modal_processors, content_type)

#                 if processor:
#                     # Generate content description
#                     description = await self._generate_query_content_description(
#                         processor, content, content_type
#                     )
#                     enhanced_parts.append(
#                         f"\nRelated {content_type} content: {description}"
#                     )
#                 else:
#                     # If no appropriate processor, use basic description
#                     basic_desc = str(content)[:200]
#                     enhanced_parts.append(
#                         f"\nRelated {content_type} content: {basic_desc}"
#                     )

#             except Exception as e:
#                 self.logger.error(f"Error processing multimodal content: {str(e)}")
#                 # Continue processing other content
#                 continue

#         enhanced_query = "\n".join(enhanced_parts)
#         enhanced_query += PROMPTS["QUERY_ENHANCEMENT_SUFFIX"]

#         self.logger.info("Multimodal query content processing completed")
#         return enhanced_query

#     async def _generate_query_content_description(
#         self, processor, content: Dict[str, Any], content_type: str
#     ) -> str:
#         """
#         Generate content description for query

#         Args:
#             processor: Multimodal processor
#             content: Content data
#             content_type: Content type

#         Returns:
#             str: Content description
#         """
#         try:
#             if content_type == "image":
#                 return await self._describe_image_for_query(processor, content)
#             elif content_type == "table":
#                 return await self._describe_table_for_query(processor, content)
#             elif content_type == "equation":
#                 return await self._describe_equation_for_query(processor, content)
#             else:
#                 return await self._describe_generic_for_query(
#                     processor, content, content_type
#                 )

#         except Exception as e:
#             self.logger.error(f"Error generating {content_type} description: {str(e)}")
#             return f"{content_type} content: {str(content)[:100]}"

#     async def _describe_image_for_query(
#         self, processor, content: Dict[str, Any]
#     ) -> str:
#         """Generate image description for query"""
#         image_path = content.get("img_path")
#         captions = content.get("image_caption", content.get("img_caption", []))
#         footnotes = content.get("image_footnote", content.get("img_footnote", []))

#         if image_path and Path(image_path).exists():
#             # If image exists, use vision model to generate description
#             image_base64 = processor._encode_image_to_base64(image_path)
#             if image_base64:
#                 prompt = PROMPTS["QUERY_IMAGE_DESCRIPTION"]
#                 description = await processor.modal_caption_func(
#                     prompt,
#                     image_data=image_base64,
#                     system_prompt=PROMPTS["QUERY_IMAGE_ANALYST_SYSTEM"],
#                 )
#                 return description

#         # If image doesn't exist or processing failed, use existing information
#         parts = []
#         if image_path:
#             parts.append(f"Image path: {image_path}")
#         if captions:
#             parts.append(f"Image captions: {', '.join(captions)}")
#         if footnotes:
#             parts.append(f"Image footnotes: {', '.join(footnotes)}")

#         return "; ".join(parts) if parts else "Image content information incomplete"

#     async def _describe_table_for_query(
#         self, processor, content: Dict[str, Any]
#     ) -> str:
#         """Generate table description for query"""
#         table_data = content.get("table_data", "")
#         table_caption = content.get("table_caption", "")

#         prompt = PROMPTS["QUERY_TABLE_ANALYSIS"].format(
#             table_data=table_data, table_caption=table_caption
#         )

#         description = await processor.modal_caption_func(
#             prompt, system_prompt=PROMPTS["QUERY_TABLE_ANALYST_SYSTEM"]
#         )

#         return description

#     async def _describe_equation_for_query(
#         self, processor, content: Dict[str, Any]
#     ) -> str:
#         """Generate equation description for query"""
#         latex = content.get("latex", "")
#         equation_caption = content.get("equation_caption", "")

#         prompt = PROMPTS["QUERY_EQUATION_ANALYSIS"].format(
#             latex=latex, equation_caption=equation_caption
#         )

#         description = await processor.modal_caption_func(
#             prompt, system_prompt=PROMPTS["QUERY_EQUATION_ANALYST_SYSTEM"]
#         )

#         return description

#     async def _describe_generic_for_query(
#         self, processor, content: Dict[str, Any], content_type: str
#     ) -> str:
#         """Generate generic content description for query"""
#         content_str = str(content)

#         prompt = PROMPTS["QUERY_GENERIC_ANALYSIS"].format(
#             content_type=content_type, content_str=content_str
#         )

#         description = await processor.modal_caption_func(
#             prompt,
#             system_prompt=PROMPTS["QUERY_GENERIC_ANALYST_SYSTEM"].format(
#                 content_type=content_type
#             ),
#         )

#         return description

#     async def _process_image_paths_for_vlm(self, prompt: str) -> tuple[str, int]:
#         """
#         Process image paths in prompt, keeping original paths and adding VLM markers

#         Args:
#             prompt: Original prompt

#         Returns:
#             tuple: (processed prompt, image count)
#         """
#         enhanced_prompt = prompt
#         images_processed = 0

#         # Initialize image cache
#         self._current_images_base64 = []

#         # Enhanced regex pattern for matching image paths
#         # Matches only the path ending with image file extensions
#         image_path_pattern = (
#             r"Image Path:\s*([^\r\n]*?\.(?:jpg|jpeg|png|gif|bmp|webp|tiff|tif))"
#         )

#         # First, let's see what matches we find
#         matches = re.findall(image_path_pattern, prompt)
#         self.logger.info(f"Found {len(matches)} image path matches in prompt")

#         def replace_image_path(match):
#             nonlocal images_processed

#             image_path = match.group(1).strip()
#             self.logger.debug(f"Processing image path: '{image_path}'")

#             # Validate path format (basic check)
#             if not image_path or len(image_path) < 3:
#                 self.logger.warning(f"Invalid image path format: {image_path}")
#                 return match.group(0)  # Keep original

#             # Use utility function to validate image file
#             self.logger.debug(f"Calling validate_image_file for: {image_path}")
#             is_valid = validate_image_file(image_path)
#             self.logger.debug(f"Validation result for {image_path}: {is_valid}")

#             if not is_valid:
#                 self.logger.warning(f"Image validation failed for: {image_path}")
#                 return match.group(0)  # Keep original if validation fails

#             try:
#                 # Encode image to base64 using utility function
#                 self.logger.debug(f"Attempting to encode image: {image_path}")
#                 image_base64 = encode_image_to_base64(image_path)
#                 if image_base64:
#                     images_processed += 1
#                     # Save base64 to instance variable for later use
#                     self._current_images_base64.append(image_base64)

#                     # Keep original path info and add VLM marker
#                     result = f"Image Path: {image_path}\n[VLM_IMAGE_{images_processed}]"
#                     self.logger.debug(
#                         f"Successfully processed image {images_processed}: {image_path}"
#                     )
#                     return result
#                 else:
#                     self.logger.error(f"Failed to encode image: {image_path}")
#                     return match.group(0)  # Keep original if encoding failed

#             except Exception as e:
#                 self.logger.error(f"Failed to process image {image_path}: {e}")
#                 return match.group(0)  # Keep original

#         # Execute replacement
#         enhanced_prompt = re.sub(
#             image_path_pattern, replace_image_path, enhanced_prompt
#         )

#         return enhanced_prompt, images_processed

#     def _build_vlm_messages_with_images(
#         self, enhanced_prompt: str, user_query: str
#     ) -> List[Dict]:
#         """
#         Build VLM message format, using markers to correspond images with text positions

#         Args:
#             enhanced_prompt: Enhanced prompt with image markers
#             user_query: User query

#         Returns:
#             List[Dict]: VLM message format
#         """
#         images_base64 = getattr(self, "_current_images_base64", [])

#         if not images_base64:
#             # Pure text mode
#             return [
#                 {
#                     "role": "user",
#                     "content": f"Context:\n{enhanced_prompt}\n\nUser Question: {user_query}",
#                 }
#             ]

#         # Build multimodal content
#         content_parts = []

#         # Split text at image markers and insert images
#         text_parts = enhanced_prompt.split("[VLM_IMAGE_")

#         for i, text_part in enumerate(text_parts):
#             if i == 0:
#                 # First text part
#                 if text_part.strip():
#                     content_parts.append({"type": "text", "text": text_part})
#             else:
#                 # Find marker number and insert corresponding image
#                 marker_match = re.match(r"(\d+)\](.*)", text_part, re.DOTALL)
#                 if marker_match:
#                     image_num = (
#                         int(marker_match.group(1)) - 1
#                     )  # Convert to 0-based index
#                     remaining_text = marker_match.group(2)

#                     # Insert corresponding image
#                     if 0 <= image_num < len(images_base64):
#                         content_parts.append(
#                             {
#                                 "type": "image_url",
#                                 "image_url": {
#                                     "url": f"data:image/jpeg;base64,{images_base64[image_num]}"
#                                 },
#                             }
#                         )

#                     # Insert remaining text
#                     if remaining_text.strip():
#                         content_parts.append({"type": "text", "text": remaining_text})

#         # Add user question
#         content_parts.append(
#             {
#                 "type": "text",
#                 "text": f"\n\nUser Question: {user_query}\n\nPlease answer based on the context and images provided.",
#             }
#         )

#         return [
#             {
#                 "role": "system",
#                 "content": "You are a helpful assistant that can analyze both text and image content to provide comprehensive answers.",
#             },
#             {"role": "user", "content": content_parts},
#         ]

#     async def _call_vlm_with_multimodal_content(self, messages: List[Dict]) -> str:
#         """
#         Call VLM to process multimodal content

#         Args:
#             messages: VLM message format

#         Returns:
#             str: VLM response result
#         """
#         try:
#             user_message = messages[1]
#             content = user_message["content"]
#             system_prompt = messages[0]["content"]

#             if isinstance(content, str):
#                 # Pure text mode
#                 result = await self.vision_model_func(
#                     content, system_prompt=system_prompt
#                 )
#             else:
#                 # Multimodal mode - pass complete messages directly to VLM
#                 result = await self.vision_model_func(
#                     "",  # Empty prompt since we're using messages format
#                     messages=messages,
#                 )

#             return result

#         except Exception as e:
#             self.logger.error(f"VLM call failed: {e}")
#             raise

#     # Synchronous versions of query methods
#     def query(self, query: str, mode: str = "mix", **kwargs) -> str:
#         """
#         Synchronous version of pure text query

#         Args:
#             query: Query text
#             mode: Query mode ("local", "global", "hybrid", "naive", "mix", "bypass")
#             **kwargs: Other query parameters, will be passed to QueryParam
#                 - vlm_enhanced: bool, default True when vision_model_func is available.
#                   If True, will parse image paths in retrieved context and replace them
#                   with base64 encoded images for VLM processing.

#         Returns:
#             str: Query result
#         """
#         loop = always_get_an_event_loop()
#         return loop.run_until_complete(self.aquery(query, mode=mode, **kwargs))

#     def query_with_multimodal(
#         self,
#         query: str,
#         multimodal_content: List[Dict[str, Any]] = None,
#         mode: str = "mix",
#         **kwargs,
#     ) -> str:
#         """
#         Synchronous version of multimodal query

#         Args:
#             query: Base query text
#             multimodal_content: List of multimodal content, each element contains:
#                 - type: Content type ("image", "table", "equation", etc.)
#                 - Other fields depend on type (e.g., img_path, table_data, latex, etc.)
#             mode: Query mode ("local", "global", "hybrid", "naive", "mix", "bypass")
#             **kwargs: Other query parameters, will be passed to QueryParam

#         Returns:
#             str: Query result
#         """
#         loop = always_get_an_event_loop()
#         return loop.run_until_complete(
#             self.aquery_with_multimodal(query, multimodal_content, mode=mode, **kwargs)
#         )

"""
Query functionality for RAGAnything - ENHANCED VERSION

Contains all query-related methods for text and multimodal queries,
plus query improvement and dual-LLM verification capabilities.
"""

import json
import hashlib
import re
import asyncio
from typing import Dict, List, Any
from pathlib import Path
from lightrag import QueryParam
from lightrag.utils import always_get_an_event_loop
from raganything.prompt import PROMPTS
from raganything.utils import (
    get_processor_for_type,
    encode_image_to_base64,
    validate_image_file,
)

# Import new enhancement modules
from raganything.query_improvement import QueryImprovementMixin
from raganything.verification import DualLLMVerificationMixin
from raganything.streaming import StreamingQueryMixin


class QueryMixin(QueryImprovementMixin, DualLLMVerificationMixin, StreamingQueryMixin):
    """
    QueryMixin class containing query functionality for RAGAnything

    Enhanced with:
    - Query improvement (rewriting, expansion, decomposition)
    - Dual-LLM verification system
    - Answer modification based on feedback
    - Real-time streaming with verification support
    """

    def _generate_multimodal_cache_key(
        self, query: str, multimodal_content: List[Dict[str, Any]], mode: str, **kwargs
    ) -> str:
        """
        Generate cache key for multimodal query

        Args:
            query: Base query text
            multimodal_content: List of multimodal content
            mode: Query mode
            **kwargs: Additional parameters

        Returns:
            str: Cache key hash
        """
        # Create a normalized representation of the query parameters
        cache_data = {
            "query": query.strip(),
            "mode": mode,
        }

        # Normalize multimodal content for stable caching
        normalized_content = []
        if multimodal_content:
            for item in multimodal_content:
                if isinstance(item, dict):
                    normalized_item = {}
                    for key, value in item.items():
                        # For file paths, use basename to make cache more portable
                        if key in [
                            "img_path",
                            "image_path",
                            "file_path",
                        ] and isinstance(value, str):
                            normalized_item[key] = Path(value).name
                        # For large content, create a hash instead of storing directly
                        elif (
                            key in ["table_data", "table_body"]
                            and isinstance(value, str)
                            and len(value) > 200
                        ):
                            normalized_item[f"{key}_hash"] = hashlib.md5(
                                value.encode()
                            ).hexdigest()
                        else:
                            normalized_item[key] = value
                    normalized_content.append(normalized_item)
                else:
                    normalized_content.append(item)

        cache_data["multimodal_content"] = normalized_content

        # Add relevant kwargs to cache data
        relevant_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "stream",
                "response_type",
                "top_k",
                "max_tokens",
                "temperature",
            ]
        }
        cache_data.update(relevant_kwargs)

        # Generate hash from the cache data
        cache_str = json.dumps(cache_data, sort_keys=True, ensure_ascii=False)
        cache_hash = hashlib.md5(cache_str.encode()).hexdigest()

        return f"multimodal_query:{cache_hash}"

    async def aquery(self, query: str, mode: str = "mix", **kwargs) -> str:
        """
        Pure text query with optional query improvement and verification

        Args:
            query: Query text
            mode: Query mode ("local", "global", "hybrid", "naive", "mix", "bypass")
            **kwargs: Other query parameters
                - vlm_enhanced: bool, default True when vision_model_func is available
                - enable_query_improvement: bool, override config setting
                - enable_verification: bool, override config setting
                - return_verification_info: bool, return detailed verification info

        Returns:
            str: Query result (or dict if return_verification_info=True)
        """
        if self.lightrag is None:
            raise ValueError(
                "No LightRAG instance available. Please process documents first or provide a pre-initialized LightRAG instance."
            )

        # Check override flags
        use_query_improvement = kwargs.pop('enable_query_improvement', 
                                           getattr(self.config, 'enable_query_improvement', False))
        use_verification = kwargs.pop('enable_verification', 
                                       getattr(self.config, 'enable_dual_llm_verification', False))
        return_verification_info = kwargs.pop('return_verification_info', False)
        
        original_query = query
        query_improvement_result = None
        
        # Step 1: Apply query improvement if enabled
        if use_query_improvement and hasattr(self, 'query_improver') and self.query_improver:
            self.logger.info("Applying query improvement...")
            query_improvement_result = await self._apply_query_improvement(query)
            if not query_improvement_result["improved_query"]:
                self.logger.warning("Query improvement resulted in an empty query, using original query.")
                query = original_query
            else:
                query = query_improvement_result["improved_query"]
            self.logger.info(f"Query improved: '{original_query[:50]}...' -> '{query[:50]}...'")
        
        # Check if VLM enhanced query should be used
        vlm_enhanced = kwargs.pop("vlm_enhanced", None)
        
        # Auto-determine VLM enhanced based on availability
        if vlm_enhanced is None:
            vlm_enhanced = (
                hasattr(self, "vision_model_func")
                and self.vision_model_func is not None
            )
        
        # If using VLM enhanced or verification is disabled, use existing flow
        if vlm_enhanced or not use_verification:
            # Use VLM enhanced query if enabled and available
            if (
                vlm_enhanced
                and hasattr(self, "vision_model_func")
                and self.vision_model_func
            ):
                result = await self.aquery_vlm_enhanced(query, mode=mode, **kwargs)
            elif vlm_enhanced and (
                not hasattr(self, "vision_model_func") or not self.vision_model_func
            ):
                self.logger.warning(
                    "VLM enhanced query requested but vision_model_func is not available, falling back to normal query"
                )
                # Create query parameters
                query_param = QueryParam(mode=mode, **kwargs)
                # Call LightRAG's query method
                result = await self.lightrag.aquery(query, param=query_param)
            else:
                # Create query parameters
                query_param = QueryParam(mode=mode, **kwargs)
                # Call LightRAG's query method
                result = await self.lightrag.aquery(query, param=query_param)

            # Handle None result from LightRAG
            if result is None:
                result = "I couldn't find any relevant information in the knowledge base to answer your question."

            # Return with verification info if requested
            if return_verification_info:
                return {
                    "answer": result,
                    "original_query": original_query,
                    "improved_query": query if query_improvement_result else original_query,
                    "query_improvement": query_improvement_result,
                    "verification_passed": True,
                    "verification_score": 10.0,
                    "modification_attempts": 0
                }

            self.logger.info("Query completed")
            return result
        
        # Step 2: Generate with verification if enabled
        if use_verification and hasattr(self, 'answer_verifier') and self.answer_verifier:
            self.logger.info("Using dual-LLM verification...")

            # Get context without final answer
            query_param = QueryParam(mode=mode, only_need_context=True, **kwargs)
            context = await self.lightrag.aquery(query, param=query_param)

            # Check if context is None or empty
            if context is None or (isinstance(context, str) and not context.strip()):
                self.logger.warning("No context retrieved from knowledge base")
                no_context_answer = "I couldn't find any relevant information in the knowledge base to answer your question."

                if return_verification_info:
                    return {
                        "answer": no_context_answer,
                        "original_query": original_query,
                        "improved_query": query if query_improvement_result else original_query,
                        "query_improvement": query_improvement_result,
                        "verification_passed": False,
                        "verification_score": 0.0,
                        "modification_attempts": 0,
                        "verification_history": []
                    }
                return no_context_answer

            # Generate with verification
            verification_result = await self._generate_with_verification(
                query=query,
                context=context,
                original_query=original_query
            )

            if return_verification_info:
                return {
                    "answer": verification_result["answer"],
                    "original_query": original_query,
                    "improved_query": query if query_improvement_result else original_query,
                    "query_improvement": query_improvement_result,
                    "verification_passed": verification_result["verification_passed"],
                    "verification_score": verification_result["verification_score"],
                    "modification_attempts": verification_result["modification_attempts"],
                    "verification_history": verification_result.get("verification_history", [])
                }

            self.logger.info("Verified query completed")
            return verification_result["answer"]
        
        # Fallback to normal query
        query_param = QueryParam(mode=mode, **kwargs)
        result = await self.lightrag.aquery(query, param=query_param)

        # Handle None result from LightRAG
        if result is None:
            result = "I couldn't find any relevant information in the knowledge base to answer your question."

        if return_verification_info:
            return {
                "answer": result,
                "original_query": original_query,
                "improved_query": query if query_improvement_result else original_query,
                "query_improvement": query_improvement_result,
                "verification_passed": True,
                "verification_score": 10.0,
                "modification_attempts": 0
            }

        self.logger.info("Query completed")
        return result

    async def aquery_with_multimodal(
        self,
        query: str,
        multimodal_content: List[Dict[str, Any]] = None,
        mode: str = "mix",
        **kwargs,
    ) -> str:
        """
        Multimodal query - combines text and multimodal content for querying

        Args:
            query: Base query text
            multimodal_content: List of multimodal content
            mode: Query mode
            **kwargs: Other query parameters

        Returns:
            str: Query result
        """
        # Ensure LightRAG is initialized
        await self._ensure_lightrag_initialized()

        self.logger.info(f"Executing multimodal query: {query[:100]}...")
        self.logger.info(f"Query mode: {mode}")

        # If no multimodal content, fallback to pure text query
        if not multimodal_content:
            self.logger.info("No multimodal content provided, executing text query")
            return await self.aquery(query, mode=mode, **kwargs)

        # Generate cache key for multimodal query
        cache_key = self._generate_multimodal_cache_key(
            query, multimodal_content, mode, **kwargs
        )

        # Check cache if available and enabled
        cached_result = None
        if (
            hasattr(self, "lightrag")
            and self.lightrag
            and hasattr(self.lightrag, "llm_response_cache")
            and self.lightrag.llm_response_cache
        ):
            if self.lightrag.llm_response_cache.global_config.get(
                "enable_llm_cache", True
            ):
                try:
                    cached_result = await self.lightrag.llm_response_cache.get_by_id(
                        cache_key
                    )
                    if cached_result and isinstance(cached_result, dict):
                        result_content = cached_result.get("return")
                        if result_content:
                            self.logger.info(
                                f"Multimodal query cache hit: {cache_key[:16]}..."
                            )
                            return result_content
                except Exception as e:
                    self.logger.debug(f"Error accessing multimodal query cache: {e}")

        # Process multimodal content to generate enhanced query text
        enhanced_query = await self._process_multimodal_query_content(
            query, multimodal_content
        )

        self.logger.info(
            f"Generated enhanced query length: {len(enhanced_query)} characters"
        )

        # Execute enhanced query
        result = await self.aquery(enhanced_query, mode=mode, **kwargs)

        # Save to cache if available and enabled
        if (
            hasattr(self, "lightrag")
            and self.lightrag
            and hasattr(self.lightrag, "llm_response_cache")
            and self.lightrag.llm_response_cache
        ):
            if self.lightrag.llm_response_cache.global_config.get(
                "enable_llm_cache", True
            ):
                try:
                    # Create cache entry for multimodal query
                    cache_entry = {
                        "return": result,
                        "cache_type": "multimodal_query",
                        "original_query": query,
                        "multimodal_content_count": len(multimodal_content),
                        "mode": mode,
                    }

                    await self.lightrag.llm_response_cache.upsert(
                        {cache_key: cache_entry}
                    )
                    self.logger.info(
                        f"Saved multimodal query result to cache: {cache_key[:16]}..."
                    )
                except Exception as e:
                    self.logger.debug(f"Error saving multimodal query to cache: {e}")

        # Ensure cache is persisted to disk
        if (
            hasattr(self, "lightrag")
            and self.lightrag
            and hasattr(self.lightrag, "llm_response_cache")
            and self.lightrag.llm_response_cache
        ):
            try:
                await self.lightrag.llm_response_cache.index_done_callback()
            except Exception as e:
                self.logger.debug(f"Error persisting multimodal query cache: {e}")

        self.logger.info("Multimodal query completed")
        return result

    async def aquery_vlm_enhanced(self, query: str, mode: str = "mix", **kwargs) -> str:
        """
        VLM enhanced query - replaces image paths in retrieved context with base64 encoded images

        Args:
            query: User query
            mode: Underlying LightRAG query mode
            **kwargs: Other query parameters

        Returns:
            str: VLM query result
        """
        # Ensure VLM is available
        if not hasattr(self, "vision_model_func") or not self.vision_model_func:
            raise ValueError(
                "VLM enhanced query requires vision_model_func. "
                "Please provide a vision model function when initializing RAGAnything."
            )

        # Ensure LightRAG is initialized
        await self._ensure_lightrag_initialized()

        self.logger.info(f"Executing VLM enhanced query: {query[:100]}...")

        # Clear previous image cache
        if hasattr(self, "_current_images_base64"):
            delattr(self, "_current_images_base64")

        # 1. Get original retrieval prompt (without generating final answer)
        self.logger.info(f"Getting raw prompt for query: {query[:100]}...")
        query_param = QueryParam(mode=mode, only_need_prompt=True, **kwargs)
        try:
            raw_prompt = await self.lightrag.aquery(query, param=query_param)
        except Exception as e:
            self.logger.error(f"Error in self.lightrag.aquery: {e}", exc_info=True)
            raw_prompt = None
        self.logger.info(f"Retrieved raw prompt: {str(raw_prompt)[:200]}...")

        if raw_prompt is None:
            self.logger.warning("raw_prompt is None, falling back to normal query (single pass)")
            query_param = QueryParam(mode=mode, **kwargs)
            return await self.lightrag.aquery(query, param=query_param)

        self.logger.debug("Retrieved raw prompt from LightRAG")

        # 2. Extract and process image paths
        enhanced_prompt, images_found = await self._process_image_paths_for_vlm(
            raw_prompt
        )

        if not images_found:
            self.logger.info("No valid images found, falling back to normal query WITHOUT re-retrieval")
            # OPTIMIZATION: Reuse the already-retrieved context instead of querying again
            # The raw_prompt already contains the full RAG context, so we can use it directly

            # Try to use the existing model function if available
            if hasattr(self.lightrag, 'llm_model_func') and self.lightrag.llm_model_func:
                try:
                    # Generate answer using the already-retrieved context
                    self.logger.info("Generating answer from cached context (avoiding re-query)")

                    # Call the LLM with the raw prompt directly
                    if asyncio.iscoroutinefunction(self.lightrag.llm_model_func):
                        result = await self.lightrag.llm_model_func(raw_prompt)
                    else:
                        result = self.lightrag.llm_model_func(raw_prompt)

                    self.logger.info("Successfully generated answer from cached context (no re-query)")
                    return result

                except Exception as e:
                    self.logger.warning(f"Failed to use cached context, falling back to re-query: {e}")
                    # Fall back to re-query if direct generation fails
                    query_param = QueryParam(mode=mode, **kwargs)
                    return await self.lightrag.aquery(query, param=query_param)
            else:
                # No model_func available, must re-query (original behavior)
                # This maintains backward compatibility
                self.logger.debug("llm_model_func not available, using standard re-query")
                query_param = QueryParam(mode=mode, **kwargs)
                return await self.lightrag.aquery(query, param=query_param)

        self.logger.info(f"Processed {images_found} images for VLM")

        # 3. Build VLM message format
        messages = self._build_vlm_messages_with_images(enhanced_prompt, query)

        # 4. Call VLM for question answering
        result = await self._call_vlm_with_multimodal_content(messages)

        self.logger.info("VLM enhanced query completed")
        return result

    # ... (rest of the existing methods remain the same) ...

    async def _process_multimodal_query_content(
        self, base_query: str, multimodal_content: List[Dict[str, Any]]
    ) -> str:
        """Process multimodal query content to generate enhanced query text"""
        self.logger.info("Starting multimodal query content processing...")

        enhanced_parts = [f"User query: {base_query}"]

        for i, content in enumerate(multimodal_content):
            content_type = content.get("type", "unknown")
            self.logger.info(
                f"Processing {i+1}/{len(multimodal_content)} multimodal content: {content_type}"
            )

            try:
                # Get appropriate processor
                processor = get_processor_for_type(self.modal_processors, content_type)

                if processor:
                    # Generate content description
                    description = await self._generate_query_content_description(
                        processor, content, content_type
                    )
                    enhanced_parts.append(
                        f"\nRelated {content_type} content: {description}"
                    )
                else:
                    # If no appropriate processor, use basic description
                    basic_desc = str(content)[:200]
                    enhanced_parts.append(
                        f"\nRelated {content_type} content: {basic_desc}"
                    )

            except Exception as e:
                self.logger.error(f"Error processing multimodal content: {str(e)}")
                continue

        enhanced_query = "\n".join(enhanced_parts)
        enhanced_query += PROMPTS["QUERY_ENHANCEMENT_SUFFIX"]

        self.logger.info("Multimodal query content processing completed")
        return enhanced_query

    async def _generate_query_content_description(
        self, processor, content: Dict[str, Any], content_type: str
    ) -> str:
        """Generate content description for query"""
        try:
            if content_type == "image":
                return await self._describe_image_for_query(processor, content)
            elif content_type == "table":
                return await self._describe_table_for_query(processor, content)
            elif content_type == "equation":
                return await self._describe_equation_for_query(processor, content)
            else:
                return await self._describe_generic_for_query(
                    processor, content, content_type
                )

        except Exception as e:
            self.logger.error(f"Error generating {content_type} description: {str(e)}")
            return f"{content_type} content: {str(content)[:100]}"

    async def _describe_image_for_query(
        self, processor, content: Dict[str, Any]
    ) -> str:
        """Generate image description for query"""
        image_path = content.get("img_path")
        captions = content.get("image_caption", content.get("img_caption", []))
        footnotes = content.get("image_footnote", content.get("img_footnote", []))

        if image_path and Path(image_path).exists():
            image_base64 = processor._encode_image_to_base64(image_path)
            if image_base64:
                prompt = PROMPTS["QUERY_IMAGE_DESCRIPTION"]
                description = await processor.modal_caption_func(
                    prompt,
                    image_data=image_base64,
                    system_prompt=PROMPTS["QUERY_IMAGE_ANALYST_SYSTEM"],
                )
                return description

        parts = []
        if image_path:
            parts.append(f"Image path: {image_path}")
        if captions:
            parts.append(f"Image captions: {', '.join(captions)}")
        if footnotes:
            parts.append(f"Image footnotes: {', '.join(footnotes)}")

        return "; ".join(parts) if parts else "Image content information incomplete"

    async def _describe_table_for_query(
        self, processor, content: Dict[str, Any]
    ) -> str:
        """Generate table description for query"""
        table_data = content.get("table_data", "")
        table_caption = content.get("table_caption", "")

        prompt = PROMPTS["QUERY_TABLE_ANALYSIS"].format(
            table_data=table_data, table_caption=table_caption
        )

        description = await processor.modal_caption_func(
            prompt, system_prompt=PROMPTS["QUERY_TABLE_ANALYST_SYSTEM"]
        )

        return description

    async def _describe_equation_for_query(
        self, processor, content: Dict[str, Any]
    ) -> str:
        """Generate equation description for query"""
        latex = content.get("latex", "")
        equation_caption = content.get("equation_caption", "")

        prompt = PROMPTS["QUERY_EQUATION_ANALYSIS"].format(
            latex=latex, equation_caption=equation_caption
        )

        description = await processor.modal_caption_func(
            prompt, system_prompt=PROMPTS["QUERY_EQUATION_ANALYST_SYSTEM"]
        )

        return description

    async def _describe_generic_for_query(
        self, processor, content: Dict[str, Any], content_type: str
    ) -> str:
        """Generate generic content description for query"""
        content_str = str(content)

        prompt = PROMPTS["QUERY_GENERIC_ANALYSIS"].format(
            content_type=content_type, content_str=content_str
        )

        description = await processor.modal_caption_func(
            prompt,
            system_prompt=PROMPTS["QUERY_GENERIC_ANALYST_SYSTEM"].format(
                content_type=content_type
            ),
        )

        return description

    async def _process_image_paths_for_vlm(self, prompt: str) -> tuple[str, int]:
        """Process image paths in prompt, keeping original paths and adding VLM markers"""
        if prompt is None:
            self.logger.warning("prompt is None in _process_image_paths_for_vlm, returning as is")
            return prompt, 0
        enhanced_prompt = prompt
        images_processed = 0

        self._current_images_base64 = []

        image_path_pattern = (
            r"Image Path:\s*([^\r\n]*?\.(?:jpg|jpeg|png|gif|bmp|webp|tiff|tif))"
        )

        matches = re.findall(image_path_pattern, prompt)
        self.logger.info(f"Found {len(matches)} image path matches in prompt")

        def replace_image_path(match):
            nonlocal images_processed

            image_path = match.group(1).strip()
            self.logger.debug(f"Processing image path: '{image_path}'")

            if not image_path or len(image_path) < 3:
                self.logger.warning(f"Invalid image path format: {image_path}")
                return match.group(0)

            self.logger.debug(f"Calling validate_image_file for: {image_path}")
            is_valid = validate_image_file(image_path)
            self.logger.debug(f"Validation result for {image_path}: {is_valid}")

            if not is_valid:
                self.logger.warning(f"Image validation failed for: {image_path}")
                return match.group(0)

            try:
                self.logger.debug(f"Attempting to encode image: {image_path}")
                image_base64 = encode_image_to_base64(image_path)
                if image_base64:
                    images_processed += 1
                    self._current_images_base64.append(image_base64)

                    result = f"Image Path: {image_path}\n[VLM_IMAGE_{images_processed}]"
                    self.logger.debug(
                        f"Successfully processed image {images_processed}: {image_path}"
                    )
                    return result
                else:
                    self.logger.error(f"Failed to encode image: {image_path}")
                    return match.group(0)

            except Exception as e:
                self.logger.error(f"Failed to process image {image_path}: {e}")
                return match.group(0)

        enhanced_prompt = re.sub(
            image_path_pattern, replace_image_path, enhanced_prompt
        )

        return enhanced_prompt, images_processed

    def _build_vlm_messages_with_images(
        self, enhanced_prompt: str, user_query: str
    ) -> List[Dict]:
        """Build VLM message format, using markers to correspond images with text positions"""
        images_base64 = getattr(self, "_current_images_base64", [])

        if not images_base64:
            return [
                {
                    "role": "user",
                    "content": f"Context:\n{enhanced_prompt}\n\nUser Question: {user_query}",
                }
            ]

        content_parts = []
        text_parts = enhanced_prompt.split("[VLM_IMAGE_")

        for i, text_part in enumerate(text_parts):
            if i == 0:
                if text_part.strip():
                    content_parts.append({"type": "text", "text": text_part})
            else:
                marker_match = re.match(r"(\d+)\](.*)", text_part, re.DOTALL)
                if marker_match:
                    image_num = int(marker_match.group(1)) - 1
                    remaining_text = marker_match.group(2)

                    if 0 <= image_num < len(images_base64):
                        content_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{images_base64[image_num]}"
                                },
                            }
                        )

                    if remaining_text.strip():
                        content_parts.append({"type": "text", "text": remaining_text})

        content_parts.append(
            {
                "type": "text",
                "text": f"\n\nUser Question: {user_query}\n\nPlease answer based on the context and images provided.",
            }
        )

        return [
            {
                "role": "system",
                "content": "You are a helpful assistant that can analyze both text and image content to provide comprehensive answers.",
            },
            {"role": "user", "content": content_parts},
        ]

    async def _call_vlm_with_multimodal_content(self, messages: List[Dict]) -> str:
        """Call VLM to process multimodal content"""
        try:
            user_message = messages[1]
            content = user_message["content"]
            system_prompt = messages[0]["content"]

            if isinstance(content, str):
                result = await self.vision_model_func(
                    content, system_prompt=system_prompt
                )
            else:
                result = await self.vision_model_func(
                    "",
                    messages=messages,
                )

            return result

        except Exception as e:
            self.logger.error(f"VLM call failed: {e}")
            raise

    # Synchronous versions of query methods
    def query(self, query: str, mode: str = "mix", **kwargs) -> str:
        """Synchronous version of pure text query"""
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, mode=mode, **kwargs))

    def query_with_multimodal(
        self,
        query: str,
        multimodal_content: List[Dict[str, Any]] = None,
        mode: str = "mix",
        **kwargs,
    ) -> str:
        """Synchronous version of multimodal query"""
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aquery_with_multimodal(query, multimodal_content, mode=mode, **kwargs)
        )