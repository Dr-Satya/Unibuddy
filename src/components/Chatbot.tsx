import { useState, useEffect, useRef } from 'react';
import iconImg from '../assets/5500_1_04.jpg';

const API_URL = 'http://127.0.0.1:9000/chat';

const stripDebug = (html) => {
  if (!html) return html;
  
  let cleaned = html;
  
  // Remove [Source: ...] debug references
  cleaned = cleaned.replace(/\[Source:[^\]]*\]/gi, '');
  
  // Remove <current_tab_state> blocks (multiple patterns to catch all variations)
  cleaned = cleaned.replace(/<current_tab_state>[\s\S]*?<\/current_tab_state>/gi, '');
  cleaned = cleaned.replace(/current_tab_state[\s\S]*?current_tab_state/gi, '');
  cleaned = cleaned.replace(new RegExp('\\u003ccurrent_tab_state\\u003e[\\s\\S]*?\\u003c\\/current_tab_state\\u003e', 'gi'), '');
  
  // Remove the specific debug block pattern from your output
  cleaned = cleaned.replace(/<current_tab_state>[\s\S]*?Open Widgets:[\s\S]*?<\/current_tab_state>/gi, '');
  
  // Remove any remaining debug information or empty lines at the end
  cleaned = cleaned.replace(/\n\s*\n\s*\n+/g, '\n\n');
  
  // Remove duplicate lines (like "His research expertise spans..." appearing twice)
  const lines = cleaned.split('\n');
  const uniqueLines = [];
  const seenLines = new Set();
  
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed && !seenLines.has(trimmed)) {
      seenLines.add(trimmed);
      uniqueLines.push(line);
    } else if (!trimmed && uniqueLines.length > 0) {
      // Only add empty lines if there is content before them
      uniqueLines.push(line);
    }
  }
  
  return uniqueLines.join('\n').trim();
};


const Chatbot = () => {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const sessionRef = useRef(null);
  const messagesContainerRef = useRef(null);

  useEffect(() => {
    if (!sessionRef.current) {
      sessionRef.current = 'sess_' + Math.random().toString(36).slice(2,9);
    }
  }, []);

  const scrollToBottom = () => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
    }
  };

  const send = async () => {
    if (!input.trim()) return;
    const userMsg = {role:'user', text: input};
    setMessages(prev => [...prev, userMsg]);
    
    // Scroll to bottom after adding user message
    setTimeout(scrollToBottom, 0);
    
    setLoading(true);
    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input, session_id: sessionRef.current })
      });
      const j = await res.json();
      console.log('API response:', j);
      const rawHtml = j.reply || '<div>Not available</div>';
      const cleaned = stripDebug(rawHtml);
      const assistantMsg = { role: 'assistant', html: cleaned, structured: j.data || null };
      setMessages(prev => [...prev, assistantMsg]);
      
      // Scroll to bottom after adding bot response
      setTimeout(scrollToBottom, 100);
    } catch (err) {
      const errMsg = { role: 'assistant', html: '<div>Error: failed to get a reply</div>' };
      setMessages(prev => [...prev, errMsg]);
      
      // Scroll to bottom after error message
      setTimeout(scrollToBottom, 0);
    } finally {
      setLoading(false);
      setInput('');
    }
  };

  return (
    <div style={{ position: 'fixed', right: 20, bottom: 20, width: 420, zIndex: 999 }}>
      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <button onClick={() => setOpen(o => !o)} style={{ background: 'none', border: 'none', padding: 0, cursor: 'pointer' }} aria-label="Open chat">
          <img src={iconImg} alt="Chat" style={{ width: 100, height: 100, borderRadius: 8 }} />
        </button>
      </div>
      {open && (
        <div style={{ marginTop: 8, border: '1px solid #ddd', borderRadius: 8, background: '#fff', boxShadow: '0 4px 12px rgba(0,0,0,0.12)' }}>
          <div ref={messagesContainerRef} style={{ maxHeight: 400, overflowY: 'auto', padding: 12, background: '#fff' }}>
            {messages.map((m, i) => (
              <div key={i} style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 12, color: '#666' }}>{m.role === 'user' ? 'You' : 'Assistant'}</div>
                <div style={{ marginTop: 6 }}>
                  <div style={{ padding: '10px 12px', borderRadius: 8, background: m.role === 'user' ? '#0066cc' : '#f7f7f7', color: m.role === 'user' ? '#fff' : '#111' }}>
                    {m.role === 'user' ? (
                      <div style={{ whiteSpace: 'pre-wrap' }}>{m.text}</div>                    ) : (
                      m.html ? (
                        <div style={{ whiteSpace: 'pre-wrap', color: '#111' }} dangerouslySetInnerHTML={{ __html: m.html }} />
                      ) : null
                    )}
                    {!m.html && m.structured && m.role !== 'user' && (
                      <div style={{ marginTop: 8 }}>
                        {m.structured.name && <div style={{ fontWeight: 700 }}>{m.structured.name}</div>}
                        {m.structured.role && <div><strong>Role:</strong> {m.structured.role}</div>}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div style={{ display: 'flex', padding: 8, borderTop: '1px solid #eee', background: '#fff' }}>
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') send(); }}
              placeholder={loading ? 'Waiting for response...' : 'Ask something...'}
              style={{ flex: 1, padding: 8, borderRadius: 4, border: '1px solid #ccc', color: '#111' }}
              disabled={loading}
            />
            <button onClick={send} disabled={loading || !input.trim()} style={{ marginLeft: 8, padding: '8px 12px' }}>
              Send
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
export default Chatbot;
