"use client";

import React, { useState, useRef, useEffect } from "react";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
};

type ChatBotProps = {
  portfolioData?: {
    optimizationMethod?: string;
    riskModel?: string;
    metrics?: {
      sharpe_ratio?: number;
      total_return?: number;
      annualized_volatility?: number;
      max_drawdown?: number;
    };
    tickers?: string[];
    portfolio_id?: string;
  };
};

// Simple markdown formatter for chat messages
function formatMessageContent(content: string): React.ReactNode {
  const lines = content.split('\n');
  const elements: React.ReactNode[] = [];
  let currentList: string[] = [];
  let listKey = 0;

  const flushList = () => {
    if (currentList.length > 0) {
      elements.push(
        <ul key={`list-${listKey++}`} className="my-2 ml-4 list-disc space-y-1">
          {currentList.map((item, idx) => (
            <li key={idx} className="text-slate-200">{item.trim()}</li>
          ))}
        </ul>
      );
      currentList = [];
    }
  };

  lines.forEach((line, idx) => {
    const trimmed = line.trim();

    // Headers
    if (trimmed.startsWith('### ')) {
      flushList();
      elements.push(
        <h3 key={idx} className="text-base font-semibold text-slate-100 mt-4 mb-2">
          {trimmed.substring(4)}
        </h3>
      );
    } else if (trimmed.startsWith('## ')) {
      flushList();
      elements.push(
        <h2 key={idx} className="text-lg font-semibold text-slate-100 mt-4 mb-2">
          {trimmed.substring(3)}
        </h2>
      );
    } else if (trimmed.startsWith('# ')) {
      flushList();
      elements.push(
        <h1 key={idx} className="text-xl font-bold text-slate-100 mt-4 mb-2">
          {trimmed.substring(2)}
        </h1>
      );
    }
    // Bullet points
    else if (trimmed.startsWith('* ') || trimmed.startsWith('- ')) {
      currentList.push(trimmed.substring(2));
    }
    // Empty line
    else if (trimmed === '') {
      flushList();
      if (idx < lines.length - 1) {
        elements.push(<br key={idx} />);
      }
    }
    // Regular text
    else {
      flushList();
      // Handle bold text (**text**) - improved regex to handle multiple bold sections
      const boldRegex = /\*\*([^*]+)\*\*/g;
      const parts: (string | React.ReactElement)[] = [];
      let lastIndex = 0;
      let match;
      let partKey = 0;

      while ((match = boldRegex.exec(trimmed)) !== null) {
        // Add text before the bold
        if (match.index > lastIndex) {
          parts.push(<span key={`text-${partKey++}`}>{trimmed.substring(lastIndex, match.index)}</span>);
        }
        // Add bold text
        parts.push(
          <strong key={`bold-${partKey++}`} className="font-semibold text-slate-100">
            {match[1]}
          </strong>
        );
        lastIndex = match.index + match[0].length;
      }

      // Add remaining text after last bold
      if (lastIndex < trimmed.length) {
        parts.push(<span key={`text-${partKey++}`}>{trimmed.substring(lastIndex)}</span>);
      }

      // If no bold text found, just use the original text
      if (parts.length === 0) {
        parts.push(<span key="text-0">{trimmed}</span>);
      }

      elements.push(
        <p key={idx} className="text-slate-200 mb-2">
          {parts}
        </p>
      );
    }
  });

  flushList(); // Flush any remaining list items

  return <div>{elements}</div>;
}

export function ChatBot({ portfolioData }: ChatBotProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      role: "assistant",
      content: "Hi! I'm your FinLove AI assistant. Ask me anything about your portfolio strategies, risk models, optimization methods, or results!",
      timestamp: new Date(),
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: inputValue.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    const question = inputValue.trim();
    setInputValue("");
    setIsLoading(true);

    try {
      // Debug: Log what we're sending
      console.log("ChatBot: Sending question with portfolio_id:", portfolioData?.portfolio_id);
      console.log("ChatBot: Full portfolioData:", portfolioData);

      const response = await fetch("/api/qa/qa", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: question,
          portfolio_id: portfolioData?.portfolio_id || null,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.answer || "I apologize, but I couldn't generate a response. Please try again.",
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Error calling QA API:", error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "I'm sorry, I encountered an error. Please make sure you've run a portfolio analysis first, or try again later.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <>
      {/* Floating Chat Button */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className="fixed bottom-6 right-6 z-50 flex h-14 w-14 items-center justify-center rounded-full bg-emerald-500 shadow-lg transition-all hover:scale-110 hover:bg-emerald-400"
          aria-label="Open chat"
        >
          <svg
            className="h-6 w-6 text-slate-950"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
            />
          </svg>
          <span className="absolute -top-1 -right-1 flex h-5 w-5 items-center justify-center rounded-full bg-rose-500 text-[10px] font-bold text-white">
            1
          </span>
        </button>
      )}

      {/* Chat Window */}
      {isOpen && (
        <div className="fixed bottom-6 right-6 z-50 flex h-[600px] w-96 flex-col rounded-2xl border border-slate-800/80 bg-slate-950/95 backdrop-blur-xl shadow-2xl">
          {/* Header */}
          <div className="flex items-center justify-between border-b border-slate-800/70 px-4 py-3">
            <div className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-500/20">
                <svg
                  className="h-4 w-4 text-emerald-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                  />
                </svg>
              </div>
              <div>
                <h3 className="text-sm font-semibold text-slate-100">
                  FinLove AI Assistant
                </h3>
                <p className="text-[10px] text-slate-400">Ask about strategies & results</p>
              </div>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              className="rounded-lg p-1.5 text-slate-400 transition-colors hover:bg-slate-800/50 hover:text-slate-200"
              aria-label="Close chat"
            >
              <svg
                className="h-5 w-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>

          {/* Messages Container */}
          <div className="flex-1 space-y-4 overflow-y-auto px-4 py-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.role === "user" ? "justify-end" : "justify-start"
                  }`}
              >
                <div
                  className={`max-w-[80%] rounded-2xl px-4 py-2.5 ${message.role === "user"
                      ? "bg-emerald-500/20 text-slate-100 border border-emerald-500/30"
                      : "bg-slate-800/50 text-slate-200 border border-slate-700/50"
                    }`}
                >
                  <div className="text-sm leading-relaxed prose prose-invert prose-sm max-w-none">
                    {formatMessageContent(message.content)}
                  </div>
                  <p className="mt-1 text-[10px] text-slate-400">
                    {message.timestamp.toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </p>
                </div>
              </div>
            ))}

            {/* Loading Indicator */}
            {isLoading && (
              <div className="flex justify-start">
                <div className="max-w-[80%] rounded-2xl bg-slate-800/50 border border-slate-700/50 px-4 py-2.5">
                  <div className="flex gap-1">
                    <div className="h-2 w-2 animate-bounce rounded-full bg-emerald-400 [animation-delay:-0.3s]"></div>
                    <div className="h-2 w-2 animate-bounce rounded-full bg-emerald-400 [animation-delay:-0.15s]"></div>
                    <div className="h-2 w-2 animate-bounce rounded-full bg-emerald-400"></div>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="border-t border-slate-800/70 p-4">
            <div className="flex items-end gap-2">
              <textarea
                ref={inputRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask about your portfolio..."
                rows={1}
                className="flex-1 resize-none rounded-lg border border-slate-700/60 bg-slate-900/60 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 outline-none focus:border-emerald-400/60 focus:ring-1 focus:ring-emerald-400/20"
                style={{
                  maxHeight: "100px",
                  minHeight: "40px",
                }}
                onInput={(e) => {
                  const target = e.target as HTMLTextAreaElement;
                  target.style.height = "auto";
                  target.style.height = `${Math.min(target.scrollHeight, 100)}px`;
                }}
              />
              <button
                onClick={handleSend}
                disabled={!inputValue.trim() || isLoading}
                className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-emerald-500 text-slate-950 transition-all hover:bg-emerald-400 disabled:opacity-50 disabled:cursor-not-allowed"
                aria-label="Send message"
              >
                {isLoading ? (
                  <svg
                    className="h-5 w-5 animate-spin"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                ) : (
                  <svg
                    className="h-5 w-5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                    />
                  </svg>
                )}
              </button>
            </div>
            <p className="mt-2 text-[10px] text-slate-500">
              Press Enter to send, Shift+Enter for new line
            </p>
          </div>
        </div>
      )}
    </>
  );
}

