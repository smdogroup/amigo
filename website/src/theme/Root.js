import React, { useEffect } from 'react';
import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';

// Client-side only component
function ThemeMessenger({ children }) {
  useEffect(() => {
    // Dynamic import to avoid SSR issues
    import('@docusaurus/theme-common').then(({ useColorMode: getColorMode }) => {
      // Get the current color mode from the DOM
      const getCurrentTheme = () => {
        return document.documentElement.getAttribute('data-theme') || 'light';
      };

      // Function to send theme to all iframes
      const sendThemeToIframes = () => {
        const theme = getCurrentTheme();
        const iframes = document.querySelectorAll('iframe');
        iframes.forEach((iframe) => {
          try {
            iframe.contentWindow.postMessage(
              {
                type: 'themeChange',
                theme: theme,
              },
              '*'
            );
          } catch (e) {
            // Ignore errors
          }
        });
      };

      // Send theme initially
      sendThemeToIframes();

      // Send theme on a slight delay to catch lazy-loaded iframes
      const timer = setTimeout(sendThemeToIframes, 100);

      // Listen for theme requests from iframes
      const handleMessage = (event) => {
        if (event.data && event.data.type === 'requestTheme') {
          const theme = getCurrentTheme();
          event.source.postMessage(
            {
              type: 'themeChange',
              theme: theme,
            },
            '*'
          );
        }
      };

      window.addEventListener('message', handleMessage);

      // Watch for theme changes using MutationObserver
      const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
          if (mutation.type === 'attributes' && mutation.attributeName === 'data-theme') {
            sendThemeToIframes();
          }
        });
      });

      observer.observe(document.documentElement, {
        attributes: true,
        attributeFilter: ['data-theme'],
      });

      return () => {
        clearTimeout(timer);
        window.removeEventListener('message', handleMessage);
        observer.disconnect();
      };
    });
  }, []);

  return <>{children}</>;
}

export default function Root({ children }) {
  // Only run on client side
  if (!ExecutionEnvironment.canUseDOM) {
    return <>{children}</>;
  }

  return <ThemeMessenger>{children}</ThemeMessenger>;
}

