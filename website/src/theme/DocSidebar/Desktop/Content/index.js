import React from 'react';
import Content from '@theme-original/DocSidebar/Desktop/Content';
import SearchBar from '@theme/SearchBar';
import styles from './styles.module.css';

export default function ContentWrapper(props) {
  return (
    <>
      <div className={styles.sidebarSearchContainer}>
        <SearchBar />
      </div>
      <Content {...props} />
    </>
  );
}

