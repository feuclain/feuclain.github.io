import { defineCollection, z } from 'astro:content';

const writing = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    category: z.enum(['ai-llm', 'algorithms', 'essays']),
    summary: z.string(),
    tags: z.array(z.string()).default([]),
    readingTime: z.string().optional(),
    cover: z.string().optional(),
    featured: z.boolean().default(false),
    draft: z.boolean().default(false),
  }),
});

const timeline = defineCollection({
  type: 'content',
  schema: z.object({
    date: z.coerce.date(),
    displayDate: z.string().optional(),
    category: z.enum(['ai-llm', 'algorithms', 'academic', 'japan', 'life']),
    title: z.string().optional(),
    location: z.string().optional(),
    images: z
      .array(
        z.object({
          src: z.string(),
          alt: z.string().default(''),
          caption: z.string().optional(),
        })
      )
      .default([]),
    links: z
      .array(
        z.object({
          label: z.string(),
          url: z.string(),
        })
      )
      .default([]),
  }),
});

export const collections = { writing, timeline };
