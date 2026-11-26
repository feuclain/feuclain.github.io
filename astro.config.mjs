import { defineConfig, passthroughImageService } from 'astro/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// Replace `site` with your own GitHub Pages URL, e.g. 'https://<username>.github.io'
// If deploying to a project repository (https://<username>.github.io/<repo>/), also set `base: '/<repo>'`
export default defineConfig({
  site: 'https://feuclain.github.io',
  image: {
    service: passthroughImageService(),
  },
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      themes: {
        light: 'github-light',
        dark: 'github-dark-dimmed',
      },
      wrap: true,
    },
  },
});
