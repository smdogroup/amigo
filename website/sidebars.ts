import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {

  docsSidebar: [
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
      label: 'User Guide',
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
      label: 'API Reference',
      items: [
        'api/overview',
        'api/component',
        'api/model',
        'api/optimizer',
      ],
    },
  ],

  tutorialsSidebar: [
    {
      type: 'category',
      label: 'Tutorials and Background',
      items: [
        {
          type: 'doc',
          id: 'tutorials/intro',
          label: 'Getting Started',
        },
        {
          type: 'category',
          label: 'Theory',
          items: [
            'tutorials/background/discretization-methods',
            'tutorials/background/direct-collocation',
            'tutorials/background/shooting-methods',
            'tutorials/background/automatic-differentiation',
            'tutorials/background/interior-point-methods',
          ],
        },
        {
          type: 'category',
          label: 'Tutorials',
          items: [
            'tutorials/cart-pole',
            'tutorials/euler_beam',
          ],
        },
      ],
    },
  ],

  examplesSidebar: [
    {
      type: 'category',
      label: 'Examples',
      items: [
        {
          type: 'doc',
          id: 'examples/overview',
          label: 'Introduction',
        },
        {
          type: 'category',
          label: 'List of Examples',
          items: [
            'tutorials/hang-glider',
            'tutorials/free-flying-robot',
            'examples/brachistochrone',
          ],
        },
      ],
    },
  ],

};

export default sidebars;
