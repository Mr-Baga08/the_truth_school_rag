/**
 * Enhanced Multi-Domain RAG Frontend with Professional Light Theme
 *
 * Features:
 * - Clean, professional light theme design
 * - Multi-domain document upload and querying
 * - Document processing status tracking
 * - Processed documents management
 * - Real-time query responses with streaming
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import 'highlight.js/styles/github.css'; // Code syntax highlighting theme
import {
  Send,
  Upload,
  FileText,
  CheckCircle,
  XCircle,
  Menu,
  X,
  Loader2,
  Trash2,
  FolderOpen,
  RefreshCw
} from 'lucide-react';

// =============================================================================
// Domain Configurations
// =============================================================================

const DOMAIN_CONFIGS = {
  medical: {
    name: 'Medical & Healthcare',
    description: 'Medical documents, research papers, clinical guidelines',
    color: '#3b82f6',
    bgColor: 'bg-blue-50',
    borderColor: 'border-blue-200',
    textColor: 'text-blue-700',
    fileTypes: ['.pdf', '.docx', '.xml', '.txt', '.doc', '.csv', '.xlsx'],
    icon: '🏥'
  },
  legal: {
    name: 'Legal & Compliance',
    description: 'Legal documents, contracts, regulations, case law',
    color: '#8b5cf6',
    bgColor: 'bg-purple-50',
    borderColor: 'border-purple-200',
    textColor: 'text-purple-700',
    fileTypes: ['.pdf', '.docx', '.txt', '.doc', '.csv', '.xlsx'],
    icon: '⚖️'
  },
  financial: {
    name: 'Financial & Analytics',
    description: 'Financial reports, analysis, market research',
    color: '#10b981',
    bgColor: 'bg-green-50',
    borderColor: 'border-green-200',
    textColor: 'text-green-700',
    fileTypes: ['.pdf', '.xlsx', '.csv', '.json', '.xls'],
    icon: '💰'
  },
  technical: {
    name: 'Technical Documentation',
    description: 'Technical docs, APIs, code, system architecture',
    color: '#f97316',
    bgColor: 'bg-orange-50',
    borderColor: 'border-orange-200',
    textColor: 'text-orange-700',
    fileTypes: ['.pdf', '.md', '.docx', '.json', '.txt', '.rst', '.csv', '.xlsx'],
    icon: '⚙️'
  },
  academic: {
    name: 'Academic Research',
    description: 'Research papers, academic publications, studies',
    color: '#6366f1',
    bgColor: 'bg-indigo-50',
    borderColor: 'border-indigo-200',
    textColor: 'text-indigo-700',
    fileTypes: ['.pdf', '.docx', '.tex', '.bib', '.txt', '.csv', '.xlsx'],
    icon: '🎓'
  }
};

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// =============================================================================
// Main Component
// =============================================================================

export default function EnhancedMultiDomainRAG() {
  // Helper function to get from localStorage with fallback
  const getFromLocalStorage = (key, defaultValue) => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : defaultValue;
    } catch (error) {
      console.error(`Error reading localStorage key "${key}":`, error);
      return defaultValue;
    }
  };

  // State Management with localStorage persistence
  const [selectedDomain, setSelectedDomain] = useState(() =>
    getFromLocalStorage('selectedDomain', 'medical')
  );
  const [currentView, setCurrentView] = useState('app'); // 'app', 'files', 'settings'
  const [processingDocs, setProcessingDocs] = useState(() =>
    getFromLocalStorage('processingDocs', [])
  );
  const [processedDocs, setProcessedDocs] = useState([]);
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState(() =>
    getFromLocalStorage('chatMessages', [])
  );
  const [isQuerying, setIsQuerying] = useState(false);
  const [error, setError] = useState(null);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [showSidebar, setShowSidebar] = useState(true);
  const [enableWebSearch, setEnableWebSearch] = useState(() =>
    getFromLocalStorage('enableWebSearch', false)
  );
  const [webSearchOnly, setWebSearchOnly] = useState(() =>
    getFromLocalStorage('webSearchOnly', false)
  );
  const [urlInput, setUrlInput] = useState('');
  const [uploadMode, setUploadMode] = useState('file'); // 'file' or 'url'
  const [fastMode, setFastMode] = useState(() =>
    getFromLocalStorage('fastMode', false)
  );
  const [enableCache, setEnableCache] = useState(() =>
    getFromLocalStorage('enableCache', true)
  );
  const [enableQueryImprovement, setEnableQueryImprovement] = useState(() =>
    getFromLocalStorage('enableQueryImprovement', true)
  );
  const [enableVerification, setEnableVerification] = useState(() =>
    getFromLocalStorage('enableVerification', true)
  );
  const [typingSpeed] = useState(0)

  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const typingQueueRef = useRef([]);
  const typingIntervalRef = useRef(null);

  // Auto-scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Persist messages to localStorage whenever they change
  useEffect(() => {
    try {
      window.localStorage.setItem('chatMessages', JSON.stringify(messages));
    } catch (error) {
      console.error('Error saving messages to localStorage:', error);
    }
  }, [messages]);

  // Persist selectedDomain to localStorage
  useEffect(() => {
    try {
      window.localStorage.setItem('selectedDomain', JSON.stringify(selectedDomain));
    } catch (error) {
      console.error('Error saving domain to localStorage:', error);
    }
  }, [selectedDomain]);

  // Persist processingDocs to localStorage
  useEffect(() => {
    try {
      window.localStorage.setItem('processingDocs', JSON.stringify(processingDocs));
    } catch (error) {
      console.error('Error saving processingDocs to localStorage:', error);
    }
  }, [processingDocs]);

  // Persist web search settings to localStorage
  useEffect(() => {
    try {
      window.localStorage.setItem('enableWebSearch', JSON.stringify(enableWebSearch));
    } catch (error) {
      console.error('Error saving enableWebSearch to localStorage:', error);
    }
  }, [enableWebSearch]);

  useEffect(() => {
    try {
      window.localStorage.setItem('webSearchOnly', JSON.stringify(webSearchOnly));
    } catch (error) {
      console.error('Error saving webSearchOnly to localStorage:', error);
    }
  }, [webSearchOnly]);

  // Persist fast mode setting to localStorage
  useEffect(() => {
    try {
      window.localStorage.setItem('fastMode', JSON.stringify(fastMode));
    } catch (error) {
      console.error('Error saving fastMode to localStorage:', error);
    }
  }, [fastMode]);

  // Persist cache setting to localStorage
  useEffect(() => {
    try {
      window.localStorage.setItem('enableCache', JSON.stringify(enableCache));
    } catch (error) {
      console.error('Error saving enableCache to localStorage:', error);
    }
  }, [enableCache]);

  // Persist query improvement setting to localStorage
  useEffect(() => {
    try {
      window.localStorage.setItem('enableQueryImprovement', JSON.stringify(enableQueryImprovement));
    } catch (error) {
      console.error('Error saving enableQueryImprovement to localStorage:', error);
    }
  }, [enableQueryImprovement]);

  // Persist verification setting to localStorage
  useEffect(() => {
    try {
      window.localStorage.setItem('enableVerification', JSON.stringify(enableVerification));
    } catch (error) {
      console.error('Error saving enableVerification to localStorage:', error);
    }
  }, [enableVerification]);

  // Persist typing speed setting to localStorage
  useEffect(() => {
    try {
      window.localStorage.setItem('typingSpeed', JSON.stringify(typingSpeed));
    } catch (error) {
      console.error('Error saving typingSpeed to localStorage:', error);
    }
  }, [typingSpeed]);

  // Fetch processed documents function with useCallback
  const fetchProcessedDocuments = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/documents?domain=${selectedDomain}`);
      if (response.ok) {
        const data = await response.json();
        const fetchedDocs = data.documents || [];

        // Merge with existing docs to avoid duplicates
        // Keep docs that exist in both, prefer fetched version for consistency
        setProcessedDocs(prev => {
          const fetchedIds = new Set(fetchedDocs.map(d => d.id));

          // Keep docs from prev that aren't in fetched (recently added via status check)
          const recentlyAdded = prev.filter(d => d.id && !fetchedIds.has(d.id));

          // Combine with fetched docs
          return [...fetchedDocs, ...recentlyAdded];
        });
      }
    } catch (err) {
      console.error('Error fetching documents:', err);
    }
  }, [selectedDomain]);

  // Check processing status function with useCallback
  const checkProcessingStatus = useCallback(async () => {
    // Update processing docs status
    const updatedProcessing = [];
    for (const doc of processingDocs) {
      try {
        const response = await fetch(`${API_BASE_URL}/status/${doc.processingId}`);
        if (response.ok) {
          const status = await response.json();
          if (status.status === 'completed') {
            // Move to processed - use processingId as id for deletion
            setProcessedDocs(prev => [...prev, {
              ...doc,
              id: doc.processingId,
              status: 'completed'
            }]);
          } else if (status.status === 'failed') {
            setError(`Processing failed for ${doc.name}: ${status.error}`);
          } else {
            updatedProcessing.push({ ...doc, status: status.status });
          }
        }
      } catch (err) {
        console.error('Error checking status:', err);
      }
    }
    setProcessingDocs(updatedProcessing);
  }, [processingDocs]);

  // Fetch processed documents on domain change
  useEffect(() => {
    fetchProcessedDocuments();
  }, [selectedDomain, fetchProcessedDocuments]);

  // Poll for document processing status
  useEffect(() => {
    const interval = setInterval(() => {
      if (processingDocs.length > 0) {
        checkProcessingStatus();
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [processingDocs, checkProcessingStatus]);

  // =============================================================================
  // API Functions
  // =============================================================================


  const handleFileUpload = async (files) => {
    if (!files || files.length === 0) return;

    setError(null);
    const newProcessingDocs = [];

    for (const file of files) {
      const fileExt = '.' + file.name.split('.').pop().toLowerCase();
      const allowedTypes = DOMAIN_CONFIGS[selectedDomain].fileTypes;

      if (!allowedTypes.includes(fileExt)) {
        setError(`File type ${fileExt} not supported for ${selectedDomain} domain. Allowed: ${allowedTypes.join(', ')}`);
        continue;
      }

      const formData = new FormData();
      formData.append('file', file);
      formData.append('domain', selectedDomain);

      try {
        const response = await fetch(`${API_BASE_URL}/upload`, {
          method: 'POST',
          body: formData
        });

        const data = await response.json();
        if (response.ok) {
          newProcessingDocs.push({
            name: file.name,
            domain: selectedDomain,
            processingId: data.processing_id,
            status: 'processing',
            uploadedAt: new Date().toISOString()
          });
        } else {
          setError(data.detail || 'Upload failed');
        }
      } catch (err) {
        console.error('Upload error:', err);
        setError(`Failed to upload ${file.name}: ${err.message}`);
      }
    }

    setProcessingDocs(prev => [...prev, ...newProcessingDocs]);
    setShowUploadModal(false);
  };

  const handleUrlUpload = async () => {
    if (!urlInput.trim()) {
      setError('Please enter a valid URL');
      return;
    }

    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/upload-url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          url: urlInput,
          domain: selectedDomain,
          convert_to_markdown: true
        })
      });

      const data = await response.json();
      if (response.ok) {
        setProcessingDocs(prev => [...prev, {
          name: urlInput,
          domain: selectedDomain,
          processingId: data.processing_id,
          status: 'processing',
          uploadedAt: new Date().toISOString()
        }]);
        setUrlInput('');
        setShowUploadModal(false);
      } else {
        setError(data.detail || 'URL upload failed');
      }
    } catch (err) {
      console.error('URL upload error:', err);
      setError(`Failed to upload URL: ${err.message}`);
    }
  };

  // Typing effect function with queue-based approach
  const startTypingEffect = useCallback((messageIndex, targetTextRef, isStreamingRef) => {
    // Clear any existing typing interval
    if (typingIntervalRef.current) {
      clearInterval(typingIntervalRef.current);
    }

    let displayedLength = 0;

    typingIntervalRef.current = setInterval(() => {
      const targetText = targetTextRef.current || '';
      const isStillStreaming = isStreamingRef.current;

      if (displayedLength < targetText.length) {
        // Add characters based on typing speed (higher = faster)
        const charsToAdd = Math.max(1, Math.floor(typingSpeed / 10));
        displayedLength = Math.min(displayedLength + charsToAdd, targetText.length);

        setMessages(prev => {
          const newMessages = [...prev];
          if (newMessages[messageIndex]) {
            newMessages[messageIndex] = {
              ...newMessages[messageIndex],
              content: targetText.substring(0, displayedLength)
            };
          }
          return newMessages;
        });
      } else if (!isStillStreaming && displayedLength >= targetText.length) {
        // If we've caught up and streaming is done, clear the interval
        clearInterval(typingIntervalRef.current);
        typingIntervalRef.current = null;
      }
    }, 30); // Update every 30ms for smoother animation
  }, [typingSpeed]);

  // Cleanup typing interval on unmount
  useEffect(() => {
    return () => {
      if (typingIntervalRef.current) {
        clearInterval(typingIntervalRef.current);
      }
    };
  }, []);

  const handleQuery = async () => {
    if (!query.trim()) return;

    setError(null);
    setIsQuerying(true);

    const userMessage = { role: 'user', content: query };
    setMessages(prev => [...prev, userMessage]);
    const currentQuery = query;
    setQuery('');

    // Create placeholder for streaming response
    const assistantMessageIndex = messages.length + 1;
    setMessages(prev => [...prev, {
      role: 'assistant',
      content: '',
      streaming: true,
      verification: null
    }]);

    // Use ref to store the full text buffer so typing effect can access it
    const fullTextBufferRef = { current: '' };
    const isStreamingRef = { current: true };
    let typingStarted = false;

    try {
      const response = await fetch(`${API_BASE_URL}/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: currentQuery,
          domain: selectedDomain,
          enable_verification: true,
          enable_web_search: enableWebSearch,
          web_search_only: webSearchOnly,
          fast_mode: fastMode,
          enable_cache: enableCache,
          enable_query_improvement: enableQueryImprovement,
          enable_verification_check: enableVerification
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Read the stream
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          break;
        }

        // Decode chunk
        buffer += decoder.decode(value, { stream: true });

        // Process complete SSE events
        const events = buffer.split('\n\n');
        buffer = events.pop() || ''; // Keep incomplete event in buffer

        for (const event of events) {
          if (!event.trim()) continue;

          const lines = event.split('\n');
          let eventType = 'message';
          let eventData = '';

          for (const line of lines) {
            if (line.startsWith('event:')) {
              eventType = line.substring(6).trim();
            } else if (line.startsWith('data:')) {
              eventData = line.substring(5).trim();
            }
          }

          if (eventData) {
            const data = JSON.parse(eventData);

            if (eventType === 'token') {
              // Add to buffer ref
              fullTextBufferRef.current += data.content;

              // Start typing effect once if speed > 0
              if (!typingStarted && typingSpeed > 0) {
                typingStarted = true;
                startTypingEffect(assistantMessageIndex, fullTextBufferRef, isStreamingRef);
              } else if (typingSpeed === 0) {
                // Instant display if typing speed is 0
                setMessages(prev => {
                  const newMessages = [...prev];
                  newMessages[assistantMessageIndex] = {
                    ...newMessages[assistantMessageIndex],
                    content: fullTextBufferRef.current
                  };
                  return newMessages;
                });
              }

            } else if (eventType === 'verification') {
              // Add verification info to message
              setMessages(prev => {
                const newMessages = [...prev];
                newMessages[assistantMessageIndex] = {
                  ...newMessages[assistantMessageIndex],
                  verification: data.content,
                  streaming: false
                };
                return newMessages;
              });

            } else if (eventType === 'done') {
              // Mark streaming as complete
              isStreamingRef.current = false;

              // Wait a bit for typing to catch up, then ensure final text is shown
              setTimeout(() => {
                if (typingIntervalRef.current) {
                  clearInterval(typingIntervalRef.current);
                  typingIntervalRef.current = null;
                }

                // Set final content and mark as complete
                setMessages(prev => {
                  const newMessages = [...prev];
                  newMessages[assistantMessageIndex] = {
                    ...newMessages[assistantMessageIndex],
                    streaming: false,
                    content: fullTextBufferRef.current
                  };
                  return newMessages;
                });
              }, typingSpeed === 0 ? 0 : 500); // Wait 500ms for typing to finish

            } else if (eventType === 'error') {
              const errorMessage = data.content.message || 'An error occurred while processing your query';
              const errorSuggestion = data.content.suggestion || '';
              setError(errorSuggestion ? `${errorMessage}\n\n${errorSuggestion}` : errorMessage);

              // Mark streaming as complete
              isStreamingRef.current = false;

              // Clear typing interval
              if (typingIntervalRef.current) {
                clearInterval(typingIntervalRef.current);
                typingIntervalRef.current = null;
              }

              // Mark message as error with helpful message
              setMessages(prev => {
                const newMessages = [...prev];
                newMessages[assistantMessageIndex] = {
                  ...newMessages[assistantMessageIndex],
                  content: fullTextBufferRef.current || errorMessage,
                  streaming: false,
                  error: true
                };
                return newMessages;
              });
              break;
            }
          }
        }
      }

    } catch (err) {
      console.error('Query error:', err);
      setError(`Query failed: ${err.message}`);

      // Clear typing interval
      if (typingIntervalRef.current) {
        clearInterval(typingIntervalRef.current);
        typingIntervalRef.current = null;
      }

      // Update message with error
      setMessages(prev => {
        const newMessages = [...prev];
        if (newMessages[assistantMessageIndex]) {
          newMessages[assistantMessageIndex] = {
            ...newMessages[assistantMessageIndex],
            content: newMessages[assistantMessageIndex].content || '[Error occurred]',
            streaming: false,
            error: true
          };
        }
        return newMessages;
      });
    } finally {
      setIsQuerying(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleQuery();
    }
  };

  const handleDeleteDocument = async (docId, docName) => {
    if (!docId) {
      console.error('Document ID is undefined');
      setError('Cannot delete document: ID is missing');
      return;
    }

    // Show confirmation dialog
    const confirmed = window.confirm(
      `Are you sure you want to delete "${docName || 'this document'}"?\n\n` +
      `This will permanently remove:\n` +
      `• All text chunks and embeddings\n` +
      `• Knowledge graph entities and relationships\n` +
      `• Vector database entries\n` +
      `• Physical files\n\n` +
      `This action cannot be undone.`
    );

    if (!confirmed) {
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/documents/${docId}`, {
        method: 'DELETE'
      });

      const data = await response.json();

      if (response.ok && data.success) {
        // Show success message with deletion details
        const report = data.report;
        const summary = report?.summary || {};

        alert(
          `✓ Document deleted successfully!\n\n` +
          `Removed from knowledge base:\n` +
          `• ${summary.chunks_deleted || 0} text chunks\n` +
          `• ${summary.entities_deleted || 0} knowledge graph entities\n` +
          `• ${summary.relationships_deleted || 0} relationships\n` +
          `• ${summary.vectors_deleted || 0} embedding vectors\n` +
          `• ${summary.files_deleted || 0} physical files\n` +
          `• ${summary.directories_deleted || 0} directories`
        );

        setProcessedDocs(prev => prev.filter(doc => doc.id !== docId));
        // Also refresh the documents list to ensure consistency
        await fetchProcessedDocuments();
      } else {
        // Show error with details if available
        const errorMsg = data.message || data.detail || 'Failed to delete document';
        const errors = data.report?.errors || [];

        setError(
          errorMsg +
          (errors.length > 0 ? `\n\nErrors: ${errors.join(', ')}` : '')
        );
      }
    } catch (err) {
      console.error('Error deleting document:', err);
      setError('Failed to delete document: ' + err.message);
    }
  };

  const clearConversation = () => {
    setMessages([]);
  };

  // =============================================================================
  // Drag and Drop Handlers
  // =============================================================================

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    handleFileUpload(e.dataTransfer.files);
  };

  // =============================================================================
  // Render Functions
  // =============================================================================

  const renderNavigation = () => (
    <nav className="bg-white border-b border-gray-200 px-6 py-3">
      <div className="flex items-center justify-between max-w-7xl mx-auto">
        <div className="flex items-center space-x-8">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">O</span>
            </div>
            <h1 className="text-xl font-semibold text-gray-800">OrgAI</h1>
            <span className="text-sm text-gray-500">/ {DOMAIN_CONFIGS[selectedDomain].name}</span>
          </div>

          <div className="flex items-center space-x-1">
            <button
              onClick={() => setCurrentView('app')}
              className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                currentView === 'app'
                  ? 'text-blue-600 bg-blue-50'
                  : 'text-gray-600 hover:text-gray-800 hover:bg-gray-50'
              }`}
            >
              App
            </button>
            <button
              onClick={() => setCurrentView('files')}
              className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                currentView === 'files'
                  ? 'text-blue-600 bg-blue-50'
                  : 'text-gray-600 hover:text-gray-800 hover:bg-gray-50'
              }`}
            >
              Files
            </button>
            <button
              onClick={() => setCurrentView('settings')}
              className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                currentView === 'settings'
                  ? 'text-blue-600 bg-blue-50'
                  : 'text-gray-600 hover:text-gray-800 hover:bg-gray-50'
              }`}
            >
              Settings
            </button>
          </div>
        </div>

        <button
          onClick={() => setShowSidebar(!showSidebar)}
          className="p-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-md"
        >
          {showSidebar ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </div>
    </nav>
  );

  const renderSidebar = () => (
    <div className={`${showSidebar ? 'w-64' : 'w-0'} transition-all duration-300 bg-gray-50 border-r border-gray-200 overflow-hidden`}>
      <div className="p-4 space-y-4">
        <div>
          <h3 className="text-xs font-semibold text-gray-500 uppercase mb-3">Domains</h3>
          <div className="space-y-1">
            {Object.entries(DOMAIN_CONFIGS).map(([key, config]) => (
              <button
                key={key}
                onClick={() => setSelectedDomain(key)}
                className={`w-full flex items-center space-x-3 px-3 py-2 rounded-lg text-sm transition-colors ${
                  selectedDomain === key
                    ? `${config.bgColor} ${config.textColor} font-medium`
                    : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                <span className="text-lg">{config.icon}</span>
                <span className="flex-1 text-left truncate">{config.name}</span>
              </button>
            ))}
          </div>
        </div>

        {processingDocs.length > 0 && (
          <div>
            <h3 className="text-xs font-semibold text-gray-500 uppercase mb-3">Processing</h3>
            <div className="space-y-2">
              {processingDocs.map((doc, idx) => (
                <div key={idx} className="flex items-center space-x-2 px-3 py-2 bg-yellow-50 rounded-lg">
                  <Loader2 className="w-4 h-4 text-yellow-600 animate-spin" />
                  <span className="text-xs text-yellow-800 truncate flex-1">{doc.name}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {processedDocs.length > 0 && (
          <div>
            <h3 className="text-xs font-semibold text-gray-500 uppercase mb-3">
              Processed Documents ({processedDocs.length})
            </h3>
            <div className="space-y-1 max-h-64 overflow-y-auto">
              {processedDocs.map((doc, idx) => (
                <div key={idx} className="flex items-center space-x-2 px-3 py-2 bg-white rounded-lg border border-gray-200 group">
                  <FileText className="w-4 h-4 text-gray-400" />
                  <span className="text-xs text-gray-700 truncate flex-1">{doc.name || `Document ${idx + 1}`}</span>
                  <button
                    onClick={() => handleDeleteDocument(doc.id, doc.name)}
                    className="opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <Trash2 className="w-3 h-3 text-gray-400 hover:text-red-600" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {messages.length > 0 && (
          <div className="pt-4 border-t border-gray-200">
            <button
              onClick={() => {
                if (window.confirm('Clear all chat history? This cannot be undone.')) {
                  setMessages([]);
                  window.localStorage.removeItem('chatMessages');
                }
              }}
              className="w-full flex items-center justify-center space-x-2 px-3 py-2 text-sm text-red-600 hover:bg-red-50 rounded-lg transition-colors"
            >
              <Trash2 className="w-4 h-4" />
              <span>Clear Chat History</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );

  const renderAppView = () => (
    <div className="flex-1 flex flex-col bg-white">
      {messages.length === 0 ? (
        <div className="flex-1 flex flex-col items-center justify-center px-4">
          <div className="text-center max-w-2xl">
            <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-6">
              <span className="text-white font-bold text-2xl">O</span>
            </div>
            <h2 className="text-3xl font-bold text-gray-800 mb-3">Welcome to OrgAI</h2>
            <p className="text-gray-600 mb-8">
              Upload documents and start chatting to get intelligent responses powered by advanced RAG technology.
            </p>

            <div className="grid grid-cols-3 gap-4 text-left">
              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="text-2xl mb-2">📄</div>
                <h3 className="font-semibold text-gray-800 mb-1">Upload Documents</h3>
                <p className="text-xs text-gray-600">Support for PDF, Word, Excel, CSV and more</p>
              </div>
              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="text-2xl mb-2">🔍</div>
                <h3 className="font-semibold text-gray-800 mb-1">Ask Questions</h3>
                <p className="text-xs text-gray-600">Get accurate answers from your documents</p>
              </div>
              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="text-2xl mb-2">⚡</div>
                <h3 className="font-semibold text-gray-800 mb-1">Multi-Domain</h3>
                <p className="text-xs text-gray-600">Optimized for medical, legal, financial and more</p>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto px-4 py-6">
          <div className="max-w-4xl mx-auto space-y-6">
            {messages.map((msg, idx) => (
              <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-2xl ${msg.role === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-800'} rounded-2xl px-4 py-3`}>
                  <div className="flex items-start space-x-2">
                    <div className="flex-1">
                      {msg.role === 'user' ? (
                        // User messages: simple text
                        <p className="text-sm whitespace-pre-wrap">
                          {msg.content}
                        </p>
                      ) : (
                        // Assistant messages: rendered markdown
                        <div className="text-sm prose prose-sm max-w-none prose-headings:mt-3 prose-headings:mb-2 prose-p:my-2 prose-ul:my-2 prose-ol:my-2 prose-li:my-0.5 prose-pre:my-2 prose-pre:bg-gray-800 prose-pre:text-gray-100">
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            rehypePlugins={[rehypeHighlight]}
                            components={{
                              // Custom styling for code blocks
                              code({ node, inline, className, children, ...props }) {
                                return inline ? (
                                  <code className="bg-gray-200 text-gray-800 px-1.5 py-0.5 rounded text-xs font-mono" {...props}>
                                    {children}
                                  </code>
                                ) : (
                                  <code className={className} {...props}>
                                    {children}
                                  </code>
                                );
                              },
                              // Custom styling for links
                              a({ node, children, ...props }) {
                                return (
                                  <a className="text-blue-600 hover:text-blue-800 underline" target="_blank" rel="noopener noreferrer" {...props}>
                                    {children}
                                  </a>
                                );
                              },
                              // Custom styling for headings
                              h1: ({ node, ...props }) => <h1 className="text-xl font-bold text-gray-900 mt-4 mb-2" {...props} />,
                              h2: ({ node, ...props }) => <h2 className="text-lg font-bold text-gray-900 mt-3 mb-2" {...props} />,
                              h3: ({ node, ...props }) => <h3 className="text-base font-semibold text-gray-900 mt-2 mb-1" {...props} />,
                              // Custom styling for lists
                              ul: ({ node, ...props }) => <ul className="list-disc list-inside space-y-1 my-2" {...props} />,
                              ol: ({ node, ...props }) => <ol className="list-decimal list-inside space-y-1 my-2" {...props} />,
                              // Custom styling for blockquotes
                              blockquote: ({ node, ...props }) => (
                                <blockquote className="border-l-4 border-gray-300 pl-4 italic text-gray-700 my-2" {...props} />
                              ),
                              // Custom styling for tables
                              table: ({ node, ...props }) => (
                                <div className="overflow-x-auto my-2">
                                  <table className="min-w-full divide-y divide-gray-200 border border-gray-200" {...props} />
                                </div>
                              ),
                              th: ({ node, ...props }) => (
                                <th className="px-3 py-2 bg-gray-50 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider border-b" {...props} />
                              ),
                              td: ({ node, ...props }) => (
                                <td className="px-3 py-2 text-sm text-gray-900 border-b" {...props} />
                              ),
                            }}
                          >
                            {msg.content}
                          </ReactMarkdown>
                          {msg.streaming && (
                            <span className="inline-block w-0.5 h-4 bg-blue-600 ml-1 animate-pulse"></span>
                          )}
                        </div>
                      )}
                    </div>
                    {msg.streaming && msg.role === 'assistant' && (
                      <Loader2 className="w-4 h-4 text-gray-400 animate-spin flex-shrink-0 mt-0.5" />
                    )}
                  </div>

                  {/* Verification Badge
                  {msg.verification && !msg.streaming && (
                    <div className={`mt-3 pt-3 border-t ${msg.role === 'user' ? 'border-blue-500' : 'border-gray-300'}`}>
                      <div className="flex items-center space-x-2 mb-2">
                        {msg.verification.passed ? (
                          <CheckCircle className="w-4 h-4 text-green-600" />
                        ) : (
                          <XCircle className="w-4 h-4 text-red-600" />
                        )}
                        <span className={`text-xs font-medium ${
                          msg.verification.passed ? 'text-green-700' : 'text-red-700'
                        }`}>
                          Verification Score: {msg.verification.score?.toFixed(1)}/10
                        </span>
                        <span className="text-xs text-gray-500">
                          ({Math.round((msg.verification.confidence || 0) * 100)}% confident)
                        </span>
                      </div>
                      {msg.verification.issues && msg.verification.issues.length > 0 && (
                        <div className="mt-2">
                          <p className="text-xs text-gray-600 font-medium mb-1">Issues found:</p>
                          <ul className="text-xs text-gray-600 space-y-0.5 list-disc list-inside">
                            {msg.verification.issues.slice(0, 3).map((issue, i) => (
                              <li key={i}>{issue}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )} */}

                  {msg.sources && msg.sources.length > 0 && (
                    <div className={`mt-3 pt-3 border-t ${msg.role === 'user' ? 'border-blue-500' : 'border-gray-300'}`}>
                      <p className="text-xs text-gray-600 mb-2">Sources:</p>
                      {msg.sources.slice(0, 3).map((source, i) => (
                        <div key={i} className="text-xs text-gray-600 mb-1">
                          • {source.file_name} (score: {source.score?.toFixed(2)})
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        </div>
      )}

      {/* Bottom Input Bar */}
      <div className="border-t border-gray-200 bg-white px-4 py-4">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setShowUploadModal(true)}
              className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <Upload className="w-4 h-4" />
              <span className="text-sm font-medium">Upload</span>
            </button>

            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder="Ask me anything or upload documents for context..."
              className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={isQuerying}
            />

            <button
              onClick={handleQuery}
              disabled={isQuerying || !query.trim()}
              className="p-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>

          {/* Web Search Options */}
          <div className="flex items-center justify-center space-x-6 mt-3">
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                checked={enableWebSearch}
                onChange={(e) => {
                  setEnableWebSearch(e.target.checked);
                  if (e.target.checked && webSearchOnly) {
                    setWebSearchOnly(false);
                  }
                }}
                className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
              />
              <span className="text-sm text-gray-700">Enhance with Web Search</span>
            </label>
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                checked={webSearchOnly}
                onChange={(e) => {
                  setWebSearchOnly(e.target.checked);
                  if (e.target.checked) {
                    setEnableWebSearch(false);
                  }
                }}
                className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
              />
              <span className="text-sm text-gray-700">Web Search Only</span>
            </label>
          </div>

          <p className="text-xs text-gray-500 mt-2 text-center">
            Press Enter to send, Shift+Enter for new line
          </p>
        </div>
      </div>
    </div>
  );

  const renderFilesView = () => (
    <div className="flex-1 overflow-y-auto p-6">
      <div className="max-w-5xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-2xl font-bold text-gray-800">Document Management</h2>
            <p className="text-gray-600">Manage your uploaded and processed documents</p>
          </div>
          <div className="flex space-x-3">
            <button
              onClick={fetchProcessedDocuments}
              className="flex items-center space-x-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              <span>Refresh</span>
            </button>
            <button
              onClick={() => setShowUploadModal(true)}
              className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <Upload className="w-4 h-4" />
              <span>Upload Documents</span>
            </button>
          </div>
        </div>

        {processingDocs.length > 0 && (
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-3">Processing Documents</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {processingDocs.map((doc, idx) => (
                <div key={idx} className="flex items-center space-x-4 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                  <Loader2 className="w-8 h-8 text-yellow-600 animate-spin" />
                  <div className="flex-1">
                    <p className="font-medium text-gray-800">{doc.name}</p>
                    <p className="text-sm text-gray-600">Processing...</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        <div>
          <h3 className="text-lg font-semibold text-gray-800 mb-3">
            Processed Documents ({processedDocs.length})
          </h3>
          {processedDocs.length === 0 ? (
            <div className="text-center py-12 bg-gray-50 rounded-lg">
              <FolderOpen className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600">No documents processed yet</p>
              <button
                onClick={() => setShowUploadModal(true)}
                className="mt-4 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Upload Your First Document
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {processedDocs.map((doc, idx) => (
                <div key={idx} className="p-4 bg-white border border-gray-200 rounded-lg hover:shadow-md transition-shadow group">
                  <div className="flex items-start justify-between mb-3">
                    <FileText className="w-8 h-8 text-blue-600" />
                    <button
                      onClick={() => handleDeleteDocument(doc.id, doc.name)}
                      className="opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:bg-gray-100 rounded"
                    >
                      <Trash2 className="w-4 h-4 text-gray-400 hover:text-red-600" />
                    </button>
                  </div>
                  <p className="font-medium text-gray-800 mb-1 truncate" title={doc.name}>{doc.name || `Document ${idx + 1}`}</p>
                  <p className="text-sm text-gray-600 mb-2">{DOMAIN_CONFIGS[doc.domain]?.name || selectedDomain}</p>
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="w-4 h-4 text-green-600" />
                    <span className="text-xs text-gray-600">Processed</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );

  const renderSettingsView = () => (
    <div className="flex-1 overflow-y-auto p-6">
      <div className="max-w-3xl mx-auto">
        <h2 className="text-2xl font-bold text-gray-800 mb-6">Settings</h2>

        <div className="space-y-6">
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Domain Configuration</h3>
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Current Domain</label>
                <select
                  value={selectedDomain}
                  onChange={(e) => setSelectedDomain(e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {Object.entries(DOMAIN_CONFIGS).map(([key, config]) => (
                    <option key={key} value={key}>{config.name}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Supported File Types</label>
                <div className="flex flex-wrap gap-2">
                  {DOMAIN_CONFIGS[selectedDomain].fileTypes.map(type => (
                    <span key={type} className="px-3 py-1 bg-gray-100 text-gray-700 text-xs rounded-full">
                      {type}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Performance Settings</h3>
            <div className="space-y-4">
              <div className="flex items-start space-x-3">
                <input
                  type="checkbox"
                  id="fastMode"
                  checked={fastMode}
                  onChange={(e) => setFastMode(e.target.checked)}
                  className="w-5 h-5 text-blue-600 border-gray-300 rounded focus:ring-blue-500 mt-0.5"
                />
                <div className="flex-1">
                  <label htmlFor="fastMode" className="block text-sm font-medium text-gray-700 cursor-pointer">
                    Fast Mode
                  </label>
                  <p className="text-xs text-gray-600 mt-1">
                    Use optimized parameters for 2-3x faster queries. Slightly reduced quality but much better performance.
                  </p>
                </div>
              </div>

              <div className="flex items-start space-x-3">
                <input
                  type="checkbox"
                  id="enableCache"
                  checked={enableCache}
                  onChange={(e) => setEnableCache(e.target.checked)}
                  className="w-5 h-5 text-blue-600 border-gray-300 rounded focus:ring-blue-500 mt-0.5"
                />
                <div className="flex-1">
                  <label htmlFor="enableCache" className="block text-sm font-medium text-gray-700 cursor-pointer">
                    Enable Query Caching
                  </label>
                  <p className="text-xs text-gray-600 mt-1">
                    Cache query results for 5 minutes. Repeated queries return instantly (100x faster).
                  </p>
                </div>
              </div>

              <div className="flex items-start space-x-3">
                <input
                  type="checkbox"
                  id="enableWebSearch"
                  checked={enableWebSearch}
                  onChange={(e) => {
                    setEnableWebSearch(e.target.checked);
                    if (e.target.checked && webSearchOnly) {
                      setWebSearchOnly(false);
                    }
                  }}
                  className="w-5 h-5 text-blue-600 border-gray-300 rounded focus:ring-blue-500 mt-0.5"
                />
                <div className="flex-1">
                  <label htmlFor="enableWebSearch" className="block text-sm font-medium text-gray-700 cursor-pointer">
                    Enhance with Web Search
                  </label>
                  <p className="text-xs text-gray-600 mt-1">
                    Augment document answers with current web search results.
                  </p>
                </div>
              </div>

              <div className="flex items-start space-x-3">
                <input
                  type="checkbox"
                  id="webSearchOnly"
                  checked={webSearchOnly}
                  onChange={(e) => {
                    setWebSearchOnly(e.target.checked);
                    if (e.target.checked) {
                      setEnableWebSearch(false);
                    }
                  }}
                  className="w-5 h-5 text-blue-600 border-gray-300 rounded focus:ring-blue-500 mt-0.5"
                />
                <div className="flex-1">
                  <label htmlFor="webSearchOnly" className="block text-sm font-medium text-gray-700 cursor-pointer">
                    Web Search Only
                  </label>
                  <p className="text-xs text-gray-600 mt-1">
                    Skip document retrieval and use only web search (useful when no documents uploaded).
                  </p>
                </div>
              </div>

              <div className="flex items-start space-x-3">
                <input
                  type="checkbox"
                  id="enableQueryImprovement"
                  checked={enableQueryImprovement}
                  onChange={(e) => setEnableQueryImprovement(e.target.checked)}
                  className="w-5 h-5 text-blue-600 border-gray-300 rounded focus:ring-blue-500 mt-0.5"
                />
                <div className="flex-1">
                  <label htmlFor="enableQueryImprovement" className="block text-sm font-medium text-gray-700 cursor-pointer">
                    Enable Query Improvement
                  </label>
                  <p className="text-xs text-gray-600 mt-1">
                    Automatically improve and expand user queries for better results. Disable for faster responses.
                  </p>
                </div>
              </div>

              <div className="flex items-start space-x-3">
                <input
                  type="checkbox"
                  id="enableVerification"
                  checked={enableVerification}
                  onChange={(e) => setEnableVerification(e.target.checked)}
                  className="w-5 h-5 text-blue-600 border-gray-300 rounded focus:ring-blue-500 mt-0.5"
                />
                <div className="flex-1">
                  <label htmlFor="enableVerification" className="block text-sm font-medium text-gray-700 cursor-pointer">
                    Enable Answer Verification
                  </label>
                  <p className="text-xs text-gray-600 mt-1">
                    Use dual-LLM verification to check answer quality and accuracy. Disable for faster responses.
                  </p>
                </div>
              </div>

    
            </div>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Actions</h3>
            <div className="space-y-3">
              <button
                onClick={clearConversation}
                className="w-full flex items-center justify-center space-x-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
              >
                <Trash2 className="w-4 h-4" />
                <span>Clear Conversation</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  // Upload Modal
  const renderUploadModal = () => {
    if (!showUploadModal) return null;

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
        <div className="bg-white rounded-xl max-w-2xl w-full p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-gray-800">Upload Documents</h2>
            <button
              onClick={() => {
                setShowUploadModal(false);
                setUploadMode('file');
                setUrlInput('');
              }}
              className="p-2 hover:bg-gray-100 rounded-lg"
            >
              <X className="w-5 h-5 text-gray-600" />
            </button>
          </div>

          {/* Mode Toggle */}
          <div className="flex items-center space-x-2 mb-6">
            <button
              onClick={() => setUploadMode('file')}
              className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
                uploadMode === 'file'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Upload File
            </button>
            <button
              onClick={() => setUploadMode('url')}
              className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
                uploadMode === 'url'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Upload from URL
            </button>
          </div>

          {uploadMode === 'file' ? (
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-xl p-12 text-center transition-colors ${
                isDragging
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-300 hover:border-gray-400'
              }`}
            >
              <Upload className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-800 mb-2">
                Drop files here or click to browse
              </h3>
              <p className="text-gray-600 mb-4">
                Supported: {DOMAIN_CONFIGS[selectedDomain].fileTypes.join(', ')}
              </p>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept={DOMAIN_CONFIGS[selectedDomain].fileTypes.join(',')}
                onChange={(e) => handleFileUpload(e.target.files)}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Select Files
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Enter URL to fetch and process
                </label>
                <input
                  type="url"
                  value={urlInput}
                  onChange={(e) => setUrlInput(e.target.value)}
                  placeholder="https://example.com/document.pdf"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      handleUrlUpload();
                    }
                  }}
                />
              </div>
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <p className="text-sm text-blue-800">
                  <strong>Supported:</strong> PDF, HTML pages (converted to markdown), and other web documents
                </p>
              </div>
              <button
                onClick={handleUrlUpload}
                disabled={!urlInput.trim()}
                className="w-full px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Fetch and Process URL
              </button>
            </div>
          )}
        </div>
      </div>
    );
  };

  // Error Display
  const renderError = () => {
    if (!error) return null;

    return (
      <div className="fixed bottom-4 right-4 bg-red-50 border border-red-200 rounded-lg p-4 max-w-md shadow-lg">
        <div className="flex items-start space-x-3">
          <XCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="text-sm text-red-800">{error}</p>
          </div>
          <button
            onClick={() => setError(null)}
            className="text-red-600 hover:text-red-800"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>
    );
  };

  // =============================================================================
  // Main Render
  // =============================================================================

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      {renderNavigation()}

      <div className="flex-1 flex overflow-hidden">
        {renderSidebar()}

        {currentView === 'app' && renderAppView()}
        {currentView === 'files' && renderFilesView()}
        {currentView === 'settings' && renderSettingsView()}
      </div>

      {renderUploadModal()}
      {renderError()}
    </div>
  );
}
