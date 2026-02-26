import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {

  docsSidebar: [
    {
      type: 'category',
      label: 'Getting started',
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
          label: 'Define a problem',
          items: [
            'manual/define-problem/components',
            'manual/define-problem/variables',
            'manual/define-problem/constraints',
            'manual/define-problem/objectives',
            'manual/define-problem/models',
          ],
        },
        'manual/set-initial-guess',
        'manual/solve-problem',
        'manual/solve-on-gpu',
        'manual/compute-flow',
        'manual/plot-solution',
      ],
    },
    {
      type: 'category',
      label: 'API reference',
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
      label: 'Tutorials and background',
      items: [
        {
          type: 'doc',
          id: 'tutorials/intro',
          label: 'Getting started',
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
          label: 'Examples browser',
        },
        {
          type: 'category',
          label: 'List of examples',
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
