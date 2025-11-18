import React from 'react';
import Content from '@theme-original/DocSidebar/Desktop/Content';
import { useLocation } from '@docusaurus/router';

export default function ContentWrapper(props) {
  const location = useLocation();
  
  return (
    <>
      {/* Search bar at top of sidebar */}
      <div style={{
        padding: '1rem 0.75rem',
        borderBottom: '1px solid var(--ifm-toc-border-color)'
      }}>
        <input
          type="text"
          placeholder="Search docs (Ctrl + /)"
          style={{
            width: '100%',
            padding: '0.5rem 0.75rem',
            border: '1px solid #d1d5db',
            borderRadius: '6px',
            fontSize: '0.9rem',
            backgroundColor: '#ffffff',
            color: '#4a4a4a'
          }}
          onFocus={(e) => {
            // You can add actual search functionality here later
            e.target.style.borderColor = '#007FA2';
            e.target.style.outline = 'none';
          }}
          onBlur={(e) => {
            e.target.style.borderColor = '#d1d5db';
          }}
        />
      </div>
      <Content {...props} />
    </>
  );
}

