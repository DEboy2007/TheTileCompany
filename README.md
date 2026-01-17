# NexHacksProject

Our project for NexHacks 2026 @ CMU

## Tech Stack

- **Next.js 16** - React framework with App Router
- **Supabase** - Backend as a Service (database, storage, etc.)
- **Tailwind CSS v4** - Utility-first CSS framework
- **TypeScript** - Type safety

## Getting Started

### 1. Install Dependencies

```bash
npm install
```

### 2. Set Up Environment Variables

Copy the example environment file:

```bash
cp .env.local.example .env.local
```

Then edit `.env.local` and add your Supabase credentials:

```env
NEXT_PUBLIC_SUPABASE_URL=your-project-url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
```

You can get these from your [Supabase project settings](https://app.supabase.com).

### 3. Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to see your app.

## Project Structure

```
.
├── app/
│   ├── globals.css          # Global styles with Tailwind directives
│   ├── layout.tsx           # Root layout component
│   └── page.tsx             # Home page
├── lib/
│   └── supabase.ts          # Supabase client configuration
├── .env.local.example       # Environment variables template
└── next.config.ts           # Next.js configuration
```

## Using Supabase

The Supabase client is initialized in `lib/supabase.ts`:

```typescript
import { supabase } from '@/lib/supabase';

// Example: Query data
const { data, error } = await supabase
  .from('your_table')
  .select('*');
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm start` - Start production server
- `npm run lint` - Run ESLint

## Next Steps

1. Create tables in your Supabase project
2. Build your application features
3. Add more pages in the `app/` directory
4. Create reusable components

## Learn More

- [Next.js Documentation](https://nextjs.org/docs)
- [Supabase Documentation](https://supabase.com/docs)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
