/**
 * TheTruthSchool - Professional AI Assistant Interface
 *
 * Features:
 * - Dark mode with elegant theme switching
 * - Claude/ChatGPT-inspired professional design
 * - Multi-domain RAG with TheTruthSchool branding
 * - Smooth animations and modern UI
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import 'highlight.js/styles/atom-one-dark.css';
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
  RefreshCw,
  Moon,
  Sun,
  Sparkles
} from 'lucide-react';

// =============================================================================
// Domain Configurations
// =============================================================================

const DOMAIN_CONFIGS = {
  medical: {
    name: 'Medical & Healthcare',
    description: 'Medical documents, research papers, clinical guidelines',
    color: '#3b82f6',
    fileTypes: ['.pdf', '.docx', '.xml', '.txt', '.doc', '.csv', '.xlsx'],
    icon: '🏥'
  },
  legal: {
    name: 'Legal & Compliance',
    description: 'Legal documents, contracts, regulations, case law',
    color: '#8b5cf6',
    fileTypes: ['.pdf', '.docx', '.txt', '.doc', '.csv', '.xlsx'],
    icon: '⚖️'
  },
  financial: {
    name: 'Financial & Analytics',
    description: 'Financial reports, analysis, market research',
    color: '#10b981',
    fileTypes: ['.pdf', '.xlsx', '.csv', '.json', '.xls'],
    icon: '💰'
  },
  technical: {
    name: 'Technical Documentation',
    description: 'Technical docs, APIs, code, system architecture',
    color: '#f97316',
    fileTypes: ['.pdf', '.md', '.docx', '.json', '.txt', '.rst', '.csv', '.xlsx'],
    icon: '⚙️'
  },
  academic: {
    name: 'Academic Research',
    description: 'Research papers, academic publications, studies',
    color: '#6366f1',
    fileTypes: ['.pdf', '.docx', '.tex', '.bib', '.txt', '.csv', '.xlsx'],
    icon: '🎓'
  }
};

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// =============================================================================
// Main Component
// =============================================================================

export default function TheTruthSchoolAI() {
  const getFromLocalStorage = (key, defaultValue) => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : defaultValue;
    } catch (error) {
      console.error(`Error reading localStorage key "${key}":`, error);
      return defaultValue;
    }
  };

  // State Management
  const [darkMode, setDarkMode] = useState(() => getFromLocalStorage('darkMode', true));
  const [selectedDomain, setSelectedDomain] = useState(() => getFromLocalStorage('selectedDomain', 'medical'));
  const [currentView, setCurrentView] = useState('app');
  const [processingDocs, setProcessingDocs] = useState(() => getFromLocalStorage('processingDocs', []));
  const [processedDocs, setProcessedDocs] = useState([]);
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState(() => getFromLocalStorage('chatMessages', []));
  const [isQuerying, setIsQuerying] = useState(false);
  const [error, setError] = useState(null);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [showSidebar, setShowSidebar] = useState(true);
  const [enableWebSearch, setEnableWebSearch] = useState(() => getFromLocalStorage('enableWebSearch', false));
  const [webSearchOnly, setWebSearchOnly] = useState(() => getFromLocalStorage('webSearchOnly', false));
  const [urlInput, setUrlInput] = useState('');
  const [uploadMode, setUploadMode] = useState('file');
  const [fastMode, setFastMode] = useState(() => getFromLocalStorage('fastMode', false));
  const [enableCache, setEnableCache] = useState(() => getFromLocalStorage('enableCache', true));
  const [enableQueryImprovement, setEnableQueryImprovement] = useState(() => getFromLocalStorage('enableQueryImprovement', true));
  const [enableVerification, setEnableVerification] = useState(() => getFromLocalStorage('enableVerification', true));
  const [typingSpeed] = useState(0);

  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const typingIntervalRef = useRef(null);

  // Theme classes based on dark mode
  const theme = {
    bg: darkMode ? 'bg-[#0D0D0D]' : 'bg-white',
    bgSecondary: darkMode ? 'bg-[#171717]' : 'bg-gray-50',
    bgTertiary: darkMode ? 'bg-[#252525]' : 'bg-white',
    text: darkMode ? 'text-gray-100' : 'text-gray-900',
    textSecondary: darkMode ? 'text-gray-400' : 'text-gray-600',
    textMuted: darkMode ? 'text-gray-500' : 'text-gray-500',
    border: darkMode ? 'border-gray-800' : 'border-gray-200',
    borderLight: darkMode ? 'border-gray-700' : 'border-gray-300',
    hover: darkMode ? 'hover:bg-[#252525]' : 'hover:bg-gray-100',
    active: darkMode ? 'bg-[#252525]' : 'bg-blue-50',
    userMessage: darkMode ? 'bg-blue-600' : 'bg-blue-600',
    assistantMessage: darkMode ? 'bg-[#252525]' : 'bg-gray-100',
    input: darkMode ? 'bg-[#171717] border-gray-700 text-gray-100' : 'bg-white border-gray-300 text-gray-900',
    button: darkMode ? 'bg-[#252525] hover:bg-[#2D2D2D]' : 'bg-gray-100 hover:bg-gray-200'
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Persist to localStorage
  useEffect(() => {
    try {
      window.localStorage.setItem('darkMode', JSON.stringify(darkMode));
      window.localStorage.setItem('chatMessages', JSON.stringify(messages));
      window.localStorage.setItem('selectedDomain', JSON.stringify(selectedDomain));
      window.localStorage.setItem('processingDocs', JSON.stringify(processingDocs));
      window.localStorage.setItem('enableWebSearch', JSON.stringify(enableWebSearch));
      window.localStorage.setItem('webSearchOnly', JSON.stringify(webSearchOnly));
      window.localStorage.setItem('fastMode', JSON.stringify(fastMode));
      window.localStorage.setItem('enableCache', JSON.stringify(enableCache));
      window.localStorage.setItem('enableQueryImprovement', JSON.stringify(enableQueryImprovement));
      window.localStorage.setItem('enableVerification', JSON.stringify(enableVerification));
    } catch (error) {
      console.error('Error saving to localStorage:', error);
    }
  }, [darkMode, messages, selectedDomain, processingDocs, enableWebSearch, webSearchOnly, fastMode, enableCache, enableQueryImprovement, enableVerification]);

  // Fetch processed documents
  const fetchProcessedDocuments = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/documents?domain=${selectedDomain}`);
      if (response.ok) {
        const data = await response.json();
        const fetchedDocs = data.documents || [];
        setProcessedDocs(prev => {
          const fetchedIds = new Set(fetchedDocs.map(d => d.id));
          const recentlyAdded = prev.filter(d => d.id && !fetchedIds.has(d.id));
          return [...fetchedDocs, ...recentlyAdded];
        });
      }
    } catch (err) {
      console.error('Error fetching documents:', err);
    }
  }, [selectedDomain]);

  // Check processing status
  const checkProcessingStatus = useCallback(async () => {
    const updatedProcessing = [];
    for (const doc of processingDocs) {
      try {
        const response = await fetch(`${API_BASE_URL}/status/${doc.processingId}`);
        if (response.ok) {
          const status = await response.json();
          if (status.status === 'completed') {
            setProcessedDocs(prev => [...prev, { ...doc, id: doc.processingId, status: 'completed' }]);
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

  useEffect(() => {
    fetchProcessedDocuments();
  }, [selectedDomain, fetchProcessedDocuments]);

  useEffect(() => {
    const interval = setInterval(() => {
      if (processingDocs.length > 0) {
        checkProcessingStatus();
      }
    }, 3000);
    return () => clearInterval(interval);
  }, [processingDocs, checkProcessingStatus]);

  // API Functions
  const handleFileUpload = async (files) => {
    if (!files || files.length === 0) return;
    setError(null);
    const newProcessingDocs = [];

    for (const file of files) {
      const fileExt = '.' + file.name.split('.').pop().toLowerCase();
      const allowedTypes = DOMAIN_CONFIGS[selectedDomain].fileTypes;

      if (!allowedTypes.includes(fileExt)) {
        setError(`File type ${fileExt} not supported for ${selectedDomain} domain.`);
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

  const startTypingEffect = useCallback((messageIndex, targetTextRef, isStreamingRef) => {
    if (typingIntervalRef.current) {
      clearInterval(typingIntervalRef.current);
    }

    let displayedLength = 0;

    typingIntervalRef.current = setInterval(() => {
      const targetText = targetTextRef.current || '';
      const isStillStreaming = isStreamingRef.current;

      if (displayedLength < targetText.length) {
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
        clearInterval(typingIntervalRef.current);
        typingIntervalRef.current = null;
      }
    }, 30);
  }, [typingSpeed]);

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

    const assistantMessageIndex = messages.length + 1;
    setMessages(prev => [...prev, {
      role: 'assistant',
      content: '',
      streaming: true,
      verification: null
    }]);

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

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });

        const events = buffer.split('\n\n');
        buffer = events.pop() || '';

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
              fullTextBufferRef.current += data.content;

              if (!typingStarted && typingSpeed > 0) {
                typingStarted = true;
                startTypingEffect(assistantMessageIndex, fullTextBufferRef, isStreamingRef);
              } else if (typingSpeed === 0) {
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
              isStreamingRef.current = false;

              setTimeout(() => {
                if (typingIntervalRef.current) {
                  clearInterval(typingIntervalRef.current);
                  typingIntervalRef.current = null;
                }

                setMessages(prev => {
                  const newMessages = [...prev];
                  newMessages[assistantMessageIndex] = {
                    ...newMessages[assistantMessageIndex],
                    streaming: false,
                    content: fullTextBufferRef.current
                  };
                  return newMessages;
                });
              }, typingSpeed === 0 ? 0 : 500);

            } else if (eventType === 'error') {
              const errorMessage = data.content.message || 'An error occurred';
              const errorSuggestion = data.content.suggestion || '';
              setError(errorSuggestion ? `${errorMessage}\n\n${errorSuggestion}` : errorMessage);

              isStreamingRef.current = false;

              if (typingIntervalRef.current) {
                clearInterval(typingIntervalRef.current);
                typingIntervalRef.current = null;
              }

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

      if (typingIntervalRef.current) {
        clearInterval(typingIntervalRef.current);
        typingIntervalRef.current = null;
      }

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

    const confirmed = window.confirm(
      `Are you sure you want to delete "${docName || 'this document'}"?\n\nThis action cannot be undone.`
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
        setProcessedDocs(prev => prev.filter(doc => doc.id !== docId));
        await fetchProcessedDocuments();
      } else {
        const errorMsg = data.message || data.detail || 'Failed to delete document';
        setError(errorMsg);
      }
    } catch (err) {
      console.error('Error deleting document:', err);
      setError('Failed to delete document: ' + err.message);
    }
  };

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
    <nav className={`${theme.bgTertiary} ${theme.border} border-b px-6 py-3`}>
      <div className="flex items-center justify-between max-w-7xl mx-auto">
        <div className="flex items-center space-x-8">
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <div className={`w-8 h-8 ${darkMode ? 'bg-gradient-to-br from-purple-500 to-blue-500' : 'bg-gradient-to-br from-blue-600 to-purple-600'} rounded-lg flex items-center justify-center`}>
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              <h1 className={`text-xl font-bold ${theme.text}`}>TheTruthSchool</h1>
            </div>
            <span className={`text-sm ${theme.textMuted}`}>/ {DOMAIN_CONFIGS[selectedDomain].name}</span>
          </div>

          <div className="flex items-center space-x-1">
            <button
              onClick={() => setCurrentView('app')}
              className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                currentView === 'app'
                  ? `${darkMode ? 'text-blue-400 bg-blue-900/30' : 'text-blue-600 bg-blue-50'}`
                  : `${theme.textSecondary} ${theme.hover}`
              }`}
            >
              Chat
            </button>
            <button
              onClick={() => setCurrentView('files')}
              className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                currentView === 'files'
                  ? `${darkMode ? 'text-blue-400 bg-blue-900/30' : 'text-blue-600 bg-blue-50'}`
                  : `${theme.textSecondary} ${theme.hover}`
              }`}
            >
              Files
            </button>
            <button
              onClick={() => setCurrentView('settings')}
              className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                currentView === 'settings'
                  ? `${darkMode ? 'text-blue-400 bg-blue-900/30' : 'text-blue-600 bg-blue-50'}`
                  : `${theme.textSecondary} ${theme.hover}`
              }`}
            >
              Settings
            </button>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => setDarkMode(!darkMode)}
            className={`p-2 ${theme.textSecondary} ${theme.hover} rounded-md transition-colors`}
          >
            {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          </button>
          <button
            onClick={() => setShowSidebar(!showSidebar)}
            className={`p-2 ${theme.textSecondary} ${theme.hover} rounded-md transition-colors`}
          >
            {showSidebar ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>
    </nav>
  );

  const renderSidebar = () => (
    <div className={`${showSidebar ? 'w-64' : 'w-0'} transition-all duration-300 ${theme.bgSecondary} ${theme.border} border-r overflow-hidden`}>
      <div className="p-4 space-y-4">
        <div>
          <h3 className={`text-xs font-semibold ${theme.textMuted} uppercase mb-3`}>Domains</h3>
          <div className="space-y-1">
            {Object.entries(DOMAIN_CONFIGS).map(([key, config]) => (
              <button
                key={key}
                onClick={() => setSelectedDomain(key)}
                className={`w-full flex items-center space-x-3 px-3 py-2 rounded-lg text-sm transition-colors ${
                  selectedDomain === key
                    ? `${darkMode ? 'bg-blue-900/30 text-blue-400' : 'bg-blue-50 text-blue-700'} font-medium`
                    : `${theme.textSecondary} ${theme.hover}`
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
            <h3 className={`text-xs font-semibold ${theme.textMuted} uppercase mb-3`}>Processing</h3>
            <div className="space-y-2">
              {processingDocs.map((doc, idx) => (
                <div key={idx} className={`flex items-center space-x-2 px-3 py-2 ${darkMode ? 'bg-yellow-900/20' : 'bg-yellow-50'} rounded-lg`}>
                  <Loader2 className={`w-4 h-4 ${darkMode ? 'text-yellow-400' : 'text-yellow-600'} animate-spin`} />
                  <span className={`text-xs ${darkMode ? 'text-yellow-300' : 'text-yellow-800'} truncate flex-1`}>{doc.name}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {processedDocs.length > 0 && (
          <div>
            <h3 className={`text-xs font-semibold ${theme.textMuted} uppercase mb-3`}>
              Documents ({processedDocs.length})
            </h3>
            <div className="space-y-1 max-h-64 overflow-y-auto">
              {processedDocs.map((doc, idx) => (
                <div key={idx} className={`flex items-center space-x-2 px-3 py-2 ${theme.bgTertiary} rounded-lg ${theme.border} border group`}>
                  <FileText className={`w-4 h-4 ${theme.textMuted}`} />
                  <span className={`text-xs ${theme.textSecondary} truncate flex-1`}>{doc.name || `Document ${idx + 1}`}</span>
                  <button
                    onClick={() => handleDeleteDocument(doc.id, doc.name)}
                    className="opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <Trash2 className={`w-3 h-3 ${theme.textMuted} hover:text-red-600`} />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {messages.length > 0 && (
          <div className={`pt-4 ${theme.border} border-t`}>
            <button
              onClick={() => {
                if (window.confirm('Clear all chat history? This cannot be undone.')) {
                  setMessages([]);
                  window.localStorage.removeItem('chatMessages');
                }
              }}
              className={`w-full flex items-center justify-center space-x-2 px-3 py-2 text-sm text-red-500 hover:${darkMode ? 'bg-red-900/20' : 'bg-red-50'} rounded-lg transition-colors`}
            >
              <Trash2 className="w-4 h-4" />
              <span>Clear Chat</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );

  const renderAppView = () => (
    <div className={`flex-1 flex flex-col ${theme.bg}`}>
      {messages.length === 0 ? (
        <div className="flex-1 flex flex-col items-center justify-center px-4">
          <div className="text-center max-w-2xl">
            <div className={`w-20 h-20 ${darkMode ? 'bg-gradient-to-br from-purple-500 to-blue-500' : 'bg-gradient-to-br from-blue-600 to-purple-600'} rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-lg`}>
              <Sparkles className="w-10 h-10 text-white" />
            </div>
            <h2 className={`text-4xl font-bold ${theme.text} mb-3`}>TheTruthSchool AI</h2>
            <p className={`${theme.textSecondary} mb-8 text-lg`}>
              Your intelligent assistant for document analysis and knowledge discovery
            </p>

            <div className="grid grid-cols-3 gap-4 text-left">
              <div className={`p-5 ${theme.bgSecondary} rounded-xl ${theme.border} border`}>
                <div className="text-3xl mb-3">📚</div>
                <h3 className={`font-semibold ${theme.text} mb-2`}>Smart Upload</h3>
                <p className={`text-sm ${theme.textSecondary}`}>Process PDFs, documents, and web content</p>
              </div>
              <div className={`p-5 ${theme.bgSecondary} rounded-xl ${theme.border} border`}>
                <div className="text-3xl mb-3">🧠</div>
                <h3 className={`font-semibold ${theme.text} mb-2`}>Deep Understanding</h3>
                <p className={`text-sm ${theme.textSecondary}`}>Advanced RAG with knowledge graphs</p>
              </div>
              <div className={`p-5 ${theme.bgSecondary} rounded-xl ${theme.border} border`}>
                <div className="text-3xl mb-3">✨</div>
                <h3 className={`font-semibold ${theme.text} mb-2`}>Multi-Domain</h3>
                <p className={`text-sm ${theme.textSecondary}`}>Optimized for healthcare, legal, finance & more</p>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto px-4 py-6">
          <div className="max-w-3xl mx-auto space-y-6">
            {messages.map((msg, idx) => (
              <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[80%] ${msg.role === 'user' ? 'bg-blue-600 text-white' : `${theme.assistantMessage} ${theme.text}`} rounded-2xl px-5 py-4 shadow-sm`}>
                  {msg.role === 'user' ? (
                    <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                  ) : (
                    <div className={`text-sm prose prose-sm max-w-none ${darkMode ? 'prose-invert' : ''}`}>
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        rehypePlugins={[rehypeHighlight]}
                        components={{
                          code({ node, inline, className, children, ...props }) {
                            return inline ? (
                              <code className={`${darkMode ? 'bg-gray-700 text-gray-100' : 'bg-gray-200 text-gray-800'} px-1.5 py-0.5 rounded text-xs font-mono`} {...props}>
                                {children}
                              </code>
                            ) : (
                              <code className={className} {...props}>
                                {children}
                              </code>
                            );
                          },
                          a({ node, children, ...props }) {
                            return (
                              <a className={`${darkMode ? 'text-blue-400' : 'text-blue-600'} hover:underline`} target="_blank" rel="noopener noreferrer" {...props}>
                                {children}
                              </a>
                            );
                          },
                          table: ({ node, ...props }) => (
                            <div className="overflow-x-auto my-4">
                              <table className={`min-w-full divide-y ${darkMode ? 'divide-gray-700 border-gray-700' : 'divide-gray-300 border-gray-300'} border rounded-lg`} {...props} />
                            </div>
                          ),
                          thead: ({ node, ...props }) => (
                            <thead className={darkMode ? 'bg-gray-800' : 'bg-gray-100'} {...props} />
                          ),
                          tbody: ({ node, ...props }) => (
                            <tbody className={`divide-y ${darkMode ? 'divide-gray-700 bg-gray-900' : 'divide-gray-200 bg-white'}`} {...props} />
                          ),
                          th: ({ node, ...props }) => (
                            <th className={`px-4 py-3 text-left text-xs font-bold uppercase tracking-wider ${darkMode ? 'text-gray-300 border-gray-700' : 'text-gray-700 border-gray-300'} border-r last:border-r-0`} {...props} />
                          ),
                          td: ({ node, ...props }) => (
                            <td className={`px-4 py-3 text-sm ${darkMode ? 'text-gray-300 border-gray-700' : 'text-gray-900 border-gray-200'} border-r last:border-r-0`} {...props} />
                          ),
                          tr: ({ node, ...props }) => (
                            <tr className={darkMode ? 'hover:bg-gray-800' : 'hover:bg-gray-50'} {...props} />
                          ),
                        }}
                      >
                        {msg.content}
                      </ReactMarkdown>
                    </div>
                  )}
                  {msg.streaming && msg.role === 'assistant' && (
                    <div className={`flex items-center space-x-1 ${theme.textMuted} text-sm mt-2`}>
                      <span>Thinking</span>
                      <span className="animate-pulse">...</span>
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
      <div className={`${theme.border} border-t ${theme.bgTertiary} px-4 py-4`}>
        <div className="max-w-3xl mx-auto">
          <div className="flex items-end space-x-3">
            <button
              onClick={() => setShowUploadModal(true)}
              className={`px-4 py-3 ${darkMode ? 'bg-blue-600 hover:bg-blue-700' : 'bg-blue-600 hover:bg-blue-700'} text-white rounded-xl transition-colors flex items-center space-x-2`}
            >
              <Upload className="w-4 h-4" />
              <span className="text-sm font-medium">Upload</span>
            </button>

            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder="Message TheTruthSchool..."
              className={`flex-1 px-4 py-3 ${theme.input} rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none`}
              disabled={isQuerying}
              rows={1}
              style={{ minHeight: '48px', maxHeight: '200px' }}
            />

            <button
              onClick={handleQuery}
              disabled={isQuerying || !query.trim()}
              className={`p-3 ${darkMode ? 'bg-blue-600 hover:bg-blue-700' : 'bg-blue-600 hover:bg-blue-700'} text-white rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              <Send className="w-5 h-5" />
            </button>
          </div>

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
                className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
              />
              <span className={`text-sm ${theme.textSecondary}`}>Enhance with Web Search</span>
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
                className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
              />
              <span className={`text-sm ${theme.textSecondary}`}>Web Search Only</span>
            </label>
          </div>

          <p className={`text-xs ${theme.textMuted} mt-2 text-center`}>
            Press Enter to send • Shift+Enter for new line
          </p>
        </div>
      </div>
    </div>
  );

  const renderFilesView = () => (
    <div className={`flex-1 overflow-y-auto p-6 ${theme.bg}`}>
      <div className="max-w-5xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className={`text-2xl font-bold ${theme.text}`}>Document Management</h2>
            <p className={theme.textSecondary}>Manage your uploaded and processed documents</p>
          </div>
          <div className="flex space-x-3">
            <button
              onClick={fetchProcessedDocuments}
              className={`flex items-center space-x-2 px-4 py-2 ${theme.button} ${theme.text} rounded-lg transition-colors`}
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
            <h3 className={`text-lg font-semibold ${theme.text} mb-3`}>Processing Documents</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {processingDocs.map((doc, idx) => (
                <div key={idx} className={`flex items-center space-x-4 p-4 ${darkMode ? 'bg-yellow-900/20 border-yellow-800' : 'bg-yellow-50 border-yellow-200'} border rounded-lg`}>
                  <Loader2 className={`w-8 h-8 ${darkMode ? 'text-yellow-400' : 'text-yellow-600'} animate-spin`} />
                  <div className="flex-1">
                    <p className={`font-medium ${theme.text}`}>{doc.name}</p>
                    <p className={`text-sm ${theme.textSecondary}`}>Processing...</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        <div>
          <h3 className={`text-lg font-semibold ${theme.text} mb-3`}>
            Processed Documents ({processedDocs.length})
          </h3>
          {processedDocs.length === 0 ? (
            <div className={`text-center py-12 ${theme.bgSecondary} rounded-lg`}>
              <FolderOpen className={`w-16 h-16 ${theme.textMuted} mx-auto mb-4`} />
              <p className={theme.textSecondary}>No documents processed yet</p>
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
                <div key={idx} className={`p-4 ${theme.bgTertiary} ${theme.border} border rounded-lg hover:shadow-lg transition-all group`}>
                  <div className="flex items-start justify-between mb-3">
                    <FileText className="w-8 h-8 text-blue-600" />
                    <button
                      onClick={() => handleDeleteDocument(doc.id, doc.name)}
                      className="opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:bg-gray-700 rounded"
                    >
                      <Trash2 className={`w-4 h-4 ${theme.textMuted} hover:text-red-500`} />
                    </button>
                  </div>
                  <p className={`font-medium ${theme.text} mb-1 truncate`} title={doc.name}>{doc.name || `Document ${idx + 1}`}</p>
                  <p className={`text-sm ${theme.textSecondary} mb-2`}>{DOMAIN_CONFIGS[doc.domain]?.name || selectedDomain}</p>
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    <span className={`text-xs ${theme.textSecondary}`}>Processed</span>
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
    <div className={`flex-1 overflow-y-auto p-6 ${theme.bg}`}>
      <div className="max-w-3xl mx-auto">
        <h2 className={`text-2xl font-bold ${theme.text} mb-6`}>Settings</h2>

        <div className="space-y-6">
          <div className={`${theme.bgTertiary} ${theme.border} border rounded-lg p-6`}>
            <h3 className={`text-lg font-semibold ${theme.text} mb-4`}>Appearance</h3>
            <div className="flex items-center justify-between">
              <div>
                <label className={`block text-sm font-medium ${theme.text}`}>Theme</label>
                <p className={`text-xs ${theme.textSecondary} mt-1`}>Choose your preferred interface theme</p>
              </div>
              <button
                onClick={() => setDarkMode(!darkMode)}
                className={`px-4 py-2 ${theme.button} ${theme.text} rounded-lg transition-colors flex items-center space-x-2`}
              >
                {darkMode ? (
                  <>
                    <Sun className="w-4 h-4" />
                    <span>Light Mode</span>
                  </>
                ) : (
                  <>
                    <Moon className="w-4 h-4" />
                    <span>Dark Mode</span>
                  </>
                )}
              </button>
            </div>
          </div>

          <div className={`${theme.bgTertiary} ${theme.border} border rounded-lg p-6`}>
            <h3 className={`text-lg font-semibold ${theme.text} mb-4`}>Domain Configuration</h3>
            <div className="space-y-3">
              <div>
                <label className={`block text-sm font-medium ${theme.text} mb-2`}>Current Domain</label>
                <select
                  value={selectedDomain}
                  onChange={(e) => setSelectedDomain(e.target.value)}
                  className={`w-full px-4 py-2 ${theme.input} rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500`}
                >
                  {Object.entries(DOMAIN_CONFIGS).map(([key, config]) => (
                    <option key={key} value={key}>{config.name}</option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          <div className={`${theme.bgTertiary} ${theme.border} border rounded-lg p-6`}>
            <h3 className={`text-lg font-semibold ${theme.text} mb-4`}>Performance Settings</h3>
            <div className="space-y-4">
              <div className="flex items-start space-x-3">
                <input
                  type="checkbox"
                  id="fastMode"
                  checked={fastMode}
                  onChange={(e) => setFastMode(e.target.checked)}
                  className="w-5 h-5 text-blue-600 rounded focus:ring-blue-500 mt-0.5"
                />
                <div className="flex-1">
                  <label htmlFor="fastMode" className={`block text-sm font-medium ${theme.text} cursor-pointer`}>
                    Fast Mode
                  </label>
                  <p className={`text-xs ${theme.textSecondary} mt-1`}>
                    Use optimized parameters for 2-3x faster queries
                  </p>
                </div>
              </div>

              <div className="flex items-start space-x-3">
                <input
                  type="checkbox"
                  id="enableCache"
                  checked={enableCache}
                  onChange={(e) => setEnableCache(e.target.checked)}
                  className="w-5 h-5 text-blue-600 rounded focus:ring-blue-500 mt-0.5"
                />
                <div className="flex-1">
                  <label htmlFor="enableCache" className={`block text-sm font-medium ${theme.text} cursor-pointer`}>
                    Enable Query Caching
                  </label>
                  <p className={`text-xs ${theme.textSecondary} mt-1`}>
                    Cache results for faster repeated queries
                  </p>
                </div>
              </div>

              <div className="flex items-start space-x-3">
                <input
                  type="checkbox"
                  id="enableQueryImprovement"
                  checked={enableQueryImprovement}
                  onChange={(e) => setEnableQueryImprovement(e.target.checked)}
                  className="w-5 h-5 text-blue-600 rounded focus:ring-blue-500 mt-0.5"
                />
                <div className="flex-1">
                  <label htmlFor="enableQueryImprovement" className={`block text-sm font-medium ${theme.text} cursor-pointer`}>
                    Enable Query Improvement
                  </label>
                  <p className={`text-xs ${theme.textSecondary} mt-1`}>
                    Automatically enhance queries for better results
                  </p>
                </div>
              </div>

              <div className="flex items-start space-x-3">
                <input
                  type="checkbox"
                  id="enableVerification"
                  checked={enableVerification}
                  onChange={(e) => setEnableVerification(e.target.checked)}
                  className="w-5 h-5 text-blue-600 rounded focus:ring-blue-500 mt-0.5"
                />
                <div className="flex-1">
                  <label htmlFor="enableVerification" className={`block text-sm font-medium ${theme.text} cursor-pointer`}>
                    Enable Answer Verification
                  </label>
                  <p className={`text-xs ${theme.textSecondary} mt-1`}>
                    Verify answer quality and accuracy with dual-LLM
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderUploadModal = () => {
    if (!showUploadModal) return null;

    return (
      <div className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50 p-4 backdrop-blur-sm">
        <div className={`${theme.bgTertiary} rounded-2xl max-w-2xl w-full p-6 shadow-2xl`}>
          <div className="flex items-center justify-between mb-6">
            <h2 className={`text-2xl font-bold ${theme.text}`}>Upload Documents</h2>
            <button
              onClick={() => {
                setShowUploadModal(false);
                setUploadMode('file');
                setUrlInput('');
              }}
              className={`p-2 ${theme.hover} rounded-lg`}
            >
              <X className={`w-5 h-5 ${theme.textSecondary}`} />
            </button>
          </div>

          <div className="flex items-center space-x-2 mb-6">
            <button
              onClick={() => setUploadMode('file')}
              className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
                uploadMode === 'file'
                  ? 'bg-blue-600 text-white'
                  : `${theme.button} ${theme.text}`
              }`}
            >
              Upload File
            </button>
            <button
              onClick={() => setUploadMode('url')}
              className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
                uploadMode === 'url'
                  ? 'bg-blue-600 text-white'
                  : `${theme.button} ${theme.text}`
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
                  ? 'border-blue-500 bg-blue-500/10'
                  : `${theme.borderLight}`
              }`}
            >
              <Upload className={`w-16 h-16 ${theme.textMuted} mx-auto mb-4`} />
              <h3 className={`text-lg font-semibold ${theme.text} mb-2`}>
                Drop files here or click to browse
              </h3>
              <p className={`${theme.textSecondary} mb-4`}>
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
                <label className={`block text-sm font-medium ${theme.text} mb-2`}>
                  Enter URL to fetch and process
                </label>
                <input
                  type="url"
                  value={urlInput}
                  onChange={(e) => setUrlInput(e.target.value)}
                  placeholder="https://example.com/document.pdf"
                  className={`w-full px-4 py-3 ${theme.input} rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500`}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      handleUrlUpload();
                    }
                  }}
                />
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

  const renderError = () => {
    if (!error) return null;

    return (
      <div className={`fixed bottom-4 right-4 ${darkMode ? 'bg-red-900/90 border-red-800' : 'bg-red-50 border-red-200'} border rounded-lg p-4 max-w-md shadow-2xl backdrop-blur-sm`}>
        <div className="flex items-start space-x-3">
          <XCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className={`text-sm ${darkMode ? 'text-red-200' : 'text-red-800'}`}>{error}</p>
          </div>
          <button
            onClick={() => setError(null)}
            className="text-red-500 hover:text-red-600"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>
    );
  };

  return (
    <div className={`h-screen flex flex-col ${theme.bg}`}>
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
