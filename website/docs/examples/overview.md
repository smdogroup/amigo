---
sidebar_position: 1
---

import React, { useState } from 'react';
import Link from '@docusaurus/Link';

export const EXAMPLES = [
  {
    name: 'Brachistochrone',
    category: 'Trajectory problem',
    difficulty: 'Beginner',
    description: 'Classic calculus of variations problem that finds the fastest sliding path between two points under gravity. Demonstrates trajectory optimization with a free final time.',
    tutorial: '/amigo/docs/examples/brachistochrone',
    github: 'https://github.com/smdogroup/amigo/tree/main/examples/brachistochrone',
  },
  {
    name: 'Cart pole',
    category: 'Trajectory problem',
    difficulty: 'Intermediate',
    description: 'Swing-up and stabilization of an inverted pendulum on a sliding cart. Introduces direct transcription and component-based modeling.',
    tutorial: '/amigo/docs/tutorials/cart-pole',
    github: 'https://github.com/smdogroup/amigo/tree/main/examples/cart_pole',
  },
  {
    name: 'Hang glider',
    category: 'Trajectory problem',
    difficulty: 'Intermediate',
    description: 'Maximizes horizontal range of a hang glider flying through a thermal updraft with lift and drag aerodynamics.',
    tutorial: '/amigo/docs/tutorials/hang-glider',
    github: 'https://github.com/smdogroup/amigo/tree/main/examples/hang_glider',
    json: '/amigo/json/hang_glider_opt_data.json',
  },
  {
    name: 'Free-flying robot',
    category: 'Trajectory problem',
    difficulty: 'Advanced',
    description: 'Minimum-time planar maneuver of a free-flying robot actuated by four independent thrusters.',
    tutorial: '/amigo/docs/tutorials/free-flying-robot',
    github: 'https://github.com/smdogroup/amigo/tree/main/examples/free_flying_robot',
    json: '/amigo/json/freeflyingrobot_opt_data.json',
  },
  {
    name: 'Euler beam',
    category: 'FEA problem',
    difficulty: 'Intermediate',
    description: 'Finite element analysis of a cantilever beam under distributed loading using Euler–Bernoulli beam theory.',
    tutorial: '/amigo/docs/tutorials/euler_beam',
    github: 'https://github.com/smdogroup/amigo/tree/main/examples/euler_beam',
  },
];

export const catStyle = {
  'Trajectory problem': { background: '#dbeafe', color: '#1e40af', border: '1px solid #bfdbfe' },
  'FEA problem':        { background: '#dcfce7', color: '#166534', border: '1px solid #bbf7d0' },
};

export const diffStyle = {
  Beginner:     { background: '#f0fdf4', color: '#166534' },
  Intermediate: { background: '#eff6ff', color: '#1e40af' },
  Advanced:     { background: '#fff7ed', color: '#9a3412' },
};

export function QuickGuide() {
  const [open, setOpen] = useState(false);
  return (
    <div style={{
      border: '2px solid #1B5299',
      borderRadius: '3px',
      marginBottom: '1.5rem',
      backgroundColor: 'var(--ifm-background-surface-color)',
      overflow: 'hidden',
    }}>
      {/* Clickable header */}
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: '100%',
          display: 'flex',
          flexDirection: 'column',
          padding: '0.65rem 1rem',
          background: 'transparent',
          border: 'none',
          cursor: 'pointer',
          textAlign: 'left',
        }}
      >
        {/* Title row */}
        <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#1B5299" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
          <span style={{ fontWeight: '700', fontSize: '0.95rem', color: '#1B5299' }}>
            Quick guide to the examples table
          </span>
        </span>
        {/* Subtitle row — always visible so user can close */}
        <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingLeft: '1.65rem', marginTop: '0.35rem' }}>
          <span style={{ fontSize: '0.78rem', color: '#374151' }}>
            {open ? 'Click to collapse the quick guide.' : 'Click to unfold and see the quick guide for the table.'}
          </span>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#374151" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
            style={{ transform: open ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s ease', flexShrink: 0 }}>
            <polyline points="6 9 12 15 18 9"/>
          </svg>
        </span>

      </button>

      {/* Expandable content */}
      {open && (
        <div style={{ padding: '0 1rem 1rem 1rem', borderTop: '1px solid #d0d7de' }}>
          <p style={{ marginTop: '0.75rem', marginBottom: '0.6rem', fontSize: '0.9rem', color: 'var(--ifm-font-color-base)' }}>
            This table lists all examples available in Amigo. Here is how to use it:
          </p>
          <ul style={{ margin: '0', paddingLeft: '1.25rem', fontSize: '0.88rem', color: 'var(--ifm-font-color-base)', lineHeight: '1.9' }}>
            <li><strong>Filter buttons</strong>{' '}: Narrow the list by problem type. Click <em>All</em> to reset the view.</li>
            <li><strong>Problem</strong>{' '}: Name of the example. Clicking it takes you directly to the tutorial page.</li>
            <li><strong>Category</strong>{' '}: Problem class: <span style={{ display:'inline-block', padding:'1px 7px', borderRadius:'8px', fontSize:'0.78rem', fontWeight:'500', background:'#dbeafe', color:'#1e40af', border:'1px solid #bfdbfe' }}>Trajectory problem</span> for optimal control and dynamics, or <span style={{ display:'inline-block', padding:'1px 7px', borderRadius:'8px', fontSize:'0.78rem', fontWeight:'500', background:'#dcfce7', color:'166534', border:'1px solid #bbf7d0' }}>FEA problem</span> for structural analysis.</li>
            <li><strong>Description</strong>{' '}: A short summary of the problem formulation and objective.</li>
            <li><strong style={{ color: '#1B5299' }}>Docs</strong>{' '}: Opens the full tutorial with problem formulation, code walkthrough, and results.</li>
            <li><strong>Code</strong>{' '}: Opens the example source code on GitHub.</li>
            <li><strong style={{ color: '#d97706' }}>JSON</strong>{' '}: Downloads the serialized model data file to reproduce the example locally (available for selected examples).</li>
          </ul>
        </div>
      )}
    </div>
  );
}

export function ExamplesTable() {
  const [filter, setFilter] = useState('All');
  const categories = ['All', 'Trajectory problem', 'FEA problem'];
  const filtered = filter === 'All' ? EXAMPLES : EXAMPLES.filter(e => e.category === filter);

  return (
    <div>
      {/* Filter bar */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '1.25rem', flexWrap: 'wrap' }}>
        <span style={{ fontSize: '0.83rem', color: 'var(--ifm-color-emphasis-600)', marginRight: '2px' }}>Filter:</span>
        {categories.map(cat => (
          <button
            key={cat}
            onClick={() => setFilter(cat)}
            style={{
              padding: '4px 14px',
              borderRadius: '20px',
              border: `1px solid ${filter === cat ? '#1B5299' : 'var(--ifm-color-emphasis-300)'}`,
              backgroundColor: filter === cat ? '#1B5299' : 'transparent',
              color: filter === cat ? '#ffffff' : 'var(--ifm-font-color-base)',
              cursor: 'pointer',
              fontSize: '0.82rem',
              fontWeight: filter === cat ? '600' : '400',
              transition: 'all 0.15s ease',
            }}
          >
            {cat}
          </button>
        ))}
        <span style={{ marginLeft: 'auto', fontSize: '0.82rem', color: 'var(--ifm-color-emphasis-600)' }}>
          {filtered.length} {filtered.length === 1 ? 'result' : 'results'}
        </span>
      </div>

      {/* Table */}
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
        <thead>
          <tr style={{ borderBottom: '2px solid var(--ifm-table-border-color)' }}>
            {['Problem', 'Category', 'Description', 'Links'].map((h, i) => (
              <th key={h} style={{
                padding: '0.55rem 0.75rem',
                textAlign: i === 3 ? 'center' : 'left',
                fontWeight: '600',
                color: 'var(--ifm-heading-color, #1a1a1a)',
                fontSize: '0.8rem',
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
              }}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {filtered.map(ex => (
            <tr key={ex.name} style={{ borderBottom: '1px solid var(--ifm-table-border-color)' }}>

              {/* Name */}
              <td style={{ padding: '0.4rem 0.75rem', textAlign: 'left', fontWeight: '500', whiteSpace: 'nowrap' }}>
                <Link to={ex.tutorial} style={{ color: '#1B5299', textDecoration: 'none' }}>
                  {ex.name}
                </Link>
              </td>

              {/* Category badge */}
              <td style={{ padding: '0.4rem 0.75rem', textAlign: 'left' }}>
                <span style={{
                  display: 'inline-block', padding: '2px 9px', borderRadius: '10px',
                  fontSize: '0.78rem', fontWeight: '500', whiteSpace: 'nowrap',
                  ...(catStyle[ex.category] || {}),
                }}>
                  {ex.category}
                </span>
              </td>

              {/* Description */}
              <td style={{ padding: '0.4rem 0.75rem', textAlign: 'justify', color: 'var(--ifm-font-color-base)', lineHeight: '1.5' }}>
                {ex.description}
              </td>

              {/* Links */}
              <td style={{ padding: '0.4rem 0.75rem', textAlign: 'center' }}>
                <div style={{ display: 'flex', gap: '6px', justifyContent: 'center' }}>
                  <Link to={ex.tutorial} style={{
                    display: 'inline-block', padding: '3px 10px', borderRadius: '4px',
                    fontSize: '0.78rem', fontWeight: '500', whiteSpace: 'nowrap',
                    border: '1px solid #1B5299', color: '#1B5299',
                    textDecoration: 'none', backgroundColor: 'transparent',
                  }}>
                    Docs
                  </Link>
                  <a href={ex.github} target="_blank" rel="noopener noreferrer" style={{
                    display: 'inline-block', padding: '3px 10px', borderRadius: '4px',
                    fontSize: '0.78rem', fontWeight: '500', whiteSpace: 'nowrap',
                    border: '1px solid var(--ifm-color-emphasis-300)',
                    color: 'var(--ifm-font-color-base)',
                    textDecoration: 'none', backgroundColor: 'transparent',
                  }}>
                    Code
                  </a>
                  {ex.json && (
                    <a href={ex.json} download target="_blank" rel="noopener noreferrer" style={{
                      display: 'inline-block', padding: '3px 10px', borderRadius: '4px',
                      fontSize: '0.78rem', fontWeight: '500', whiteSpace: 'nowrap',
                      border: '1px solid #d97706',
                      color: '#d97706',
                      textDecoration: 'none', backgroundColor: 'transparent',
                    }}>
                      JSON
                    </a>
                  )}
                </div>
              </td>

            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

# Examples browser

This collection of examples is part of Amigo. Amigo gathers tools for modeling and solving optimization problems and applications. It aims to provide efficient, differentiable solvers for trajectory optimization, optimal control, and finite element analysis, both on CPU and GPU. If you want to define a problem and solve it, please check the [documentation](/amigo/docs/getting-started/introduction).

<QuickGuide />

<div style={{ borderTop: '1px solid #e5e7eb', margin: '1.25rem 0' }} />

From this page, you can find a list of examples solved with Amigo. The table below provides an overview of all available problems and allows interactive exploration and filtering by problem type.

<ExamplesTable />
