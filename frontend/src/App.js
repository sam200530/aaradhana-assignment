import React, { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// ── Zodiac symbol helper ─────────────────────────────────────────────────────
const ZODIAC_SYMBOLS = {
  Aries: '♈', Taurus: '♉', Gemini: '♊', Cancer: '♋',
  Leo: '♌', Virgo: '♍', Libra: '♎', Scorpio: '♏',
  Sagittarius: '♐', Capricorn: '♑', Aquarius: '♒', Pisces: '♓',
};

// ── Birth Details Form ────────────────────────────────────────────────────────
function BirthDetailsForm({ onSubmit, initialDetails }) {
  const [form, setForm] = useState(initialDetails || {
    name: '', date_of_birth: '', time_of_birth: '', place_of_birth: '',
  });
  const [errors, setErrors] = useState({});

  const validate = () => {
    const e = {};
    if (!form.name.trim()) e.name = 'Name is required';
    if (!form.date_of_birth) e.date_of_birth = 'Birth date is required';
    else {
      const d = new Date(form.date_of_birth);
      if (isNaN(d.getTime()) || d > new Date()) e.date_of_birth = 'Enter a valid past date';
    }
    if (!form.place_of_birth.trim()) e.place_of_birth = 'Birth place is required';
    return e;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const errs = validate();
    if (Object.keys(errs).length) { setErrors(errs); return; }
    onSubmit(form);
  };

  const field = (key, label, type = 'text', placeholder = '') => (
    <div className="field">
      <label>{label}</label>
      <input
        type={type}
        value={form[key]}
        placeholder={placeholder}
        max={type === 'date' ? new Date().toISOString().split('T')[0] : undefined}
        onChange={e => { setForm(p => ({ ...p, [key]: e.target.value })); setErrors(p => ({ ...p, [key]: '' })); }}
      />
      {errors[key] && <span className="field-error">{errors[key]}</span>}
    </div>
  );

  return (
    <form className="birth-form" onSubmit={handleSubmit}>
      <div className="birth-form-header">
        <span className="star-glyph">✦</span>
        <h2>Your Birth Details</h2>
        <p>Share where and when you arrived on Earth — the stars remember.</p>
      </div>
      {field('name', 'Your Name', 'text', 'e.g. Priya')}
      {field('date_of_birth', 'Date of Birth', 'date')}
      <div className="field">
        <label>Time of Birth <span className="optional">(optional but recommended)</span></label>
        <input
          type="time"
          value={form.time_of_birth}
          onChange={e => setForm(p => ({ ...p, time_of_birth: e.target.value }))}
        />
        <span className="field-hint">Without birth time, house positions won't be available</span>
      </div>
      {field('place_of_birth', 'Place of Birth', 'text', 'e.g. Mumbai, India')}
      <button type="submit" className="btn-primary">
        Begin Reading ✦
      </button>
    </form>
  );
}

// ── Tool Activity Card ────────────────────────────────────────────────────────
function ToolActivity({ activities }) {
  if (!activities.length) return null;
  return (
    <div className="tool-activities">
      {activities.map((a, i) => (
        <div key={i} className={`tool-chip ${a.type === 'tool_call' ? 'calling' : 'done'}`}>
          <span className="tool-dot" />
          <span className="tool-label">
            {a.type === 'tool_call'
              ? `Consulting ${a.tool_name?.replace('tool_', '').replace(/_/g, ' ')}…`
              : `✓ ${a.tool_name?.replace('tool_', '').replace(/_/g, ' ')}`}
          </span>
        </div>
      ))}
    </div>
  );
}

// ── Message Bubble ─────────────────────────────────────────────────────────────
function MessageBubble({ msg }) {
  if (msg.role === 'user') {
    return (
      <div className="bubble-row user">
        <div className="bubble user-bubble">{msg.content}</div>
      </div>
    );
  }

  return (
    <div className="bubble-row assistant">
      <div className="avatar">☽</div>
      <div className="bubble-stack">
        <ToolActivity activities={msg.toolActivities || []} />
        <div className={`bubble assistant-bubble ${msg.streaming ? 'streaming' : ''}`}>
          {msg.content || <span className="thinking-dots"><span>.</span><span>.</span><span>.</span></span>}
        </div>
      </div>
    </div>
  );
}

// ── Suggested Prompts ─────────────────────────────────────────────────────────
const SUGGESTED_PROMPTS_WITH_CHART = [
  "What does my chart reveal about my life path?",
  "Tell me about today's planetary energy for me",
  "What does my Moon sign say about my emotions?",
  "How does my Saturn placement shape my challenges?",
];

const SUGGESTED_PROMPTS_NO_CHART = [
  "What does Mercury retrograde mean?",
  "Tell me about Venus in Libra",
  "What is a Saturn return?",
  "Explain trine aspects to me",
];

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [birthDetails, setBirthDetails] = useState(null);
  const [sessionId] = useState(() => crypto.randomUUID());
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [showForm, setShowForm] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);
  const abortRef = useRef(null);

  // Load from localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem('aradhana_birth');
      if (saved) setBirthDetails(JSON.parse(saved));
      const savedHistory = localStorage.getItem(`aradhana_history_${sessionId}`);
      if (savedHistory) setMessages(JSON.parse(savedHistory));
    } catch {}
  }, []);

  // Persist messages
  useEffect(() => {
    if (messages.length) {
      localStorage.setItem(`aradhana_history_${sessionId}`, JSON.stringify(messages));
    }
  }, [messages, sessionId]);

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleBirthSubmit = (details) => {
    setBirthDetails(details);
    localStorage.setItem('aradhana_birth', JSON.stringify(details));
    setShowForm(false);
    setMessages([{
      role: 'assistant',
      content: `Namaste, ${details.name} 🙏 Your birth details are saved. I can now read your natal chart and reflect on the planetary patterns of your life. What would you like to explore?`,
      toolActivities: [],
      id: Date.now(),
    }]);
    setShowSuggestions(true);
  };

  const sendMessage = useCallback(async (text) => {
    if (!text.trim() || isStreaming) return;

    setShowSuggestions(false);
    const userMsg = { role: 'user', content: text, id: Date.now() };
    const assistantMsgId = Date.now() + 1;
    const assistantMsg = {
      role: 'assistant', content: '', toolActivities: [], streaming: true, id: assistantMsgId,
    };

    setMessages(prev => [...prev, userMsg, assistantMsg]);
    setInput('');
    setIsStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const resp = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          message: text,
          birth_details: birthDetails,
        }),
        signal: controller.signal,
      });

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const data = line.slice(6).trim();
          if (data === '[DONE]') break;

          try {
            const event = JSON.parse(data);

            setMessages(prev => prev.map(m => {
              if (m.id !== assistantMsgId) return m;

              if (event.type === 'text') {
                return { ...m, content: m.content + event.content };
              }
              if (event.type === 'tool_call' || event.type === 'tool_result') {
                const newActivities = [...m.toolActivities];
                if (event.type === 'tool_call') {
                  newActivities.push({ type: 'tool_call', tool_name: event.tool_name });
                } else {
                  // Replace last matching tool_call with done
                  const idx = newActivities.findLastIndex(
                    a => a.type === 'tool_call' && a.tool_name === event.tool_name
                  );
                  if (idx !== -1) newActivities[idx] = { type: 'tool_result', tool_name: event.tool_name };
                  else newActivities.push({ type: 'tool_result', tool_name: event.tool_name });
                }
                return { ...m, toolActivities: newActivities };
              }
              return m;
            }));
          } catch {}
        }
      }

    } catch (err) {
      if (err.name !== 'AbortError') {
        setMessages(prev => prev.map(m =>
          m.id === assistantMsgId
            ? { ...m, content: 'I seem to have lost the cosmic signal. Please try again in a moment.', streaming: false }
            : m
        ));
      }
    } finally {
      setMessages(prev => prev.map(m =>
        m.id === assistantMsgId ? { ...m, streaming: false } : m
      ));
      setIsStreaming(false);
      inputRef.current?.focus();
    }
  }, [birthDetails, isStreaming, sessionId]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setShowSuggestions(true);
    localStorage.removeItem(`aradhana_history_${sessionId}`);
  };

  const suggestions = birthDetails ? SUGGESTED_PROMPTS_WITH_CHART : SUGGESTED_PROMPTS_NO_CHART;

  return (
    <div className="app">
      {/* Cosmic background */}
      <div className="cosmos-bg" aria-hidden="true">
        <div className="cosmos-ring r1" />
        <div className="cosmos-ring r2" />
        <div className="cosmos-ring r3" />
      </div>

      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <span className="logo-glyph">☽</span>
          <span className="logo-text">Aradhana</span>
        </div>

        <div className="sidebar-section">
          <p className="sidebar-label">Your Chart</p>
          {birthDetails ? (
            <div className="chart-summary">
              <p className="chart-name">{birthDetails.name}</p>
              <p className="chart-detail">{birthDetails.date_of_birth}</p>
              <p className="chart-detail">{birthDetails.place_of_birth}</p>
              <button className="btn-ghost small" onClick={() => setShowForm(true)}>Edit Details</button>
            </div>
          ) : (
            <button className="btn-ghost" onClick={() => setShowForm(true)}>
              ✦ Enter Birth Details
            </button>
          )}
        </div>

        <div className="sidebar-section">
          <p className="sidebar-label">Planets Today</p>
          <div className="planet-pills">
            {['☉ Sun', '☽ Moon', '☿ Mercury', '♀ Venus', '♂ Mars'].map(p => (
              <span key={p} className="planet-pill">{p}</span>
            ))}
          </div>
        </div>

        <div className="sidebar-bottom">
          <button className="btn-ghost small" onClick={clearChat}>Clear Chat</button>
          <p className="disclaimer-text">
            Aradhana offers cosmic reflection, not medical, legal, or financial advice.
          </p>
        </div>
      </aside>

      {/* Main chat area */}
      <main className="chat-main">
        {/* Header */}
        <div className="chat-header">
          <div className="chat-header-inner">
            <h1 className="chat-title">
              {birthDetails ? `Reading for ${birthDetails.name}` : 'Aradhana'}
            </h1>
            <p className="chat-subtitle">Your Daily Spiritual Companion</p>
          </div>
        </div>

        {/* Messages */}
        <div className="messages-container">
          {messages.length === 0 && !showForm && (
            <div className="welcome">
              <div className="welcome-glyph">✦</div>
              <h2>Namaste</h2>
              <p>
                I am Aradhana, your guide through the cosmic language of the stars.
                Share your birth details and I will read your natal chart, explore today's planetary energies,
                or reflect on any question you carry.
              </p>
              {!birthDetails && (
                <button className="btn-primary" onClick={() => setShowForm(true)}>
                  Enter Your Birth Details ✦
                </button>
              )}
            </div>
          )}

          {showForm && (
            <div className="form-overlay">
              <BirthDetailsForm onSubmit={handleBirthSubmit} initialDetails={birthDetails} />
              {birthDetails && (
                <button className="btn-ghost small" onClick={() => setShowForm(false)}>Cancel</button>
              )}
            </div>
          )}

          {!showForm && messages.map(msg => (
            <MessageBubble key={msg.id} msg={msg} />
          ))}

          {!showForm && showSuggestions && messages.length <= 1 && (
            <div className="suggestions">
              <p className="suggestions-label">You might ask…</p>
              <div className="suggestions-grid">
                {suggestions.map((s, i) => (
                  <button key={i} className="suggestion-chip" onClick={() => sendMessage(s)}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* Input bar */}
        {!showForm && (
          <div className="input-bar">
            <div className="input-inner">
              <textarea
                ref={inputRef}
                className="chat-input"
                placeholder="Ask about your chart, today's energy, or any cosmic question…"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                rows={1}
                disabled={isStreaming}
              />
              <button
                className="send-btn"
                onClick={() => sendMessage(input)}
                disabled={!input.trim() || isStreaming}
                aria-label="Send message"
              >
                {isStreaming ? '◎' : '✦'}
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
