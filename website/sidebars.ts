import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'getting-started/introduction',
        'getting-started/installation',
      ],
    },
    {
      type: 'category',
      label: 'Manual',
      items: [
        {
          type: 'category',
          label: 'Define Problem',
          items: [
            'manual/define-problem/components',
            'manual/define-problem/variables',
            'manual/define-problem/constraints',
            'manual/define-problem/objectives',
            'manual/define-problem/models',
          ],
        },
        'manual/solve-problem',
        'manual/solve-on-gpu',
      ],
    },
    {
      type: 'category',
      label: 'Tutorials',
      items: [
        'tutorials/cart-pole',
      ],
    },
    {
      type: 'category',
      label: 'Examples Gallery',
      items: [
        'examples/overview',
        'examples/optimization',
        'examples/optimal-control',
        'examples/multidisciplinary',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: [
        'api/overview',
        'api/component',
        'api/model',
        'api/optimizer',
        'api/external-component',
      ],
    },
    {
      type: 'category',
      label: 'Developers',
      items: [
        'developers/architecture',
        'developers/contributing',
        'developers/testing',
      ],
    },
  ],
};

export default sidebars;

