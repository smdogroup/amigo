import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

export default function Home(): JSX.Element {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="A friendly Python library for multidisciplinary design optimization on high-performance computing resources">
      <main style={{padding: '4rem 2rem', maxWidth: '1000px', margin: '0 auto'}}>
        <div style={{marginBottom: '3rem', textAlign: 'center'}}>
          <Heading as="h1" style={{fontSize: '3rem', fontWeight: '600', marginBottom: '2rem', color: '#1a1a1a'}}>
            Amigo
          </Heading>
        </div>

        <div style={{fontSize: '1.05rem', lineHeight: '1.7', color: '#333333', textAlign: 'justify'}}>
          <p>
            Amigo is a Python library for solving multidisciplinary analysis and optimization problems on high-performance 
            computing systems through automatically generated C++ wrappers. All application code is written in Python and 
            automatically compiled to C++ with automatic differentiation using A2D.
          </p>

          <p>
            Multiple backend implementations are supported: Serial, OpenMP, and MPI (CUDA for Nvidia GPUs is under development). 
            User code and model construction are independent of the target backend. Amigo integrates seamlessly with OpenMDAO 
            through <code>amigo.ExternalComponent</code> and can be used as a sub-optimization component with accurate 
            post-optimality derivatives.
          </p>

          <Heading as="h2" style={{
            fontSize: '1.75rem', 
            fontWeight: '600', 
            marginTop: '2.5rem', 
            marginBottom: '1rem', 
            color: '#1a1a1a',
            paddingBottom: '0.5rem',
            borderBottom: '2px solid #e5e7eb'
          }}>
            Getting started
          </Heading>

          <p>
            To solve your first optimal control problem using <strong>Amigo</strong>, please check the{' '}
            <Link to="/docs/getting-started/introduction" style={{color: '#4063D8', textDecoration: 'none'}}>documentation</Link>, or simply try our{' '}
            <Link to="/docs/tutorials/cart-pole" style={{color: '#4063D8', textDecoration: 'none'}}>cart-pole tutorial</Link>.
          </p>
        </div>
      </main>
    </Layout>
  );
}

