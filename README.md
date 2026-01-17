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

### 2. Set Up Supabase (Optional)

You have two options for connecting to Supabase:

**Option A: Use the UI (Recommended)**
- Start the dev server and navigate to http://localhost:3000
- Click "Add Credentials" on the dashboard
- Enter your Supabase URL and API key
- Credentials are saved in your browser's localStorage

**Option B: Use Environment Variables**
- Copy `.env.local.example` to `.env.local`
- Add your Supabase credentials to `.env.local`

### 3. Run Local PostgreSQL (Optional)

Want to test with a local database instead? We've included a Docker setup:

```bash
cd database
./start.sh
```

This will spin up a PostgreSQL database with sample data. See `database/README.md` for details.

### 4. Run Development Server

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
│   └── page.tsx             # Home page with credentials manager
├── database/
│   ├── docker-compose.yml   # PostgreSQL Docker setup
│   ├── init.sql             # Database initialization script
│   ├── start.sh             # Script to start database
│   ├── stop.sh              # Script to stop database
│   └── README.md            # Database documentation
├── lib/
│   └── supabase.ts          # Supabase client configuration
├── .env.local.example       # Environment variables template
└── next.config.ts           # Next.js configuration
```

## Using Supabase

### Managing Credentials

The app includes a built-in credentials manager on the dashboard:
- **Add/Update**: Enter your Supabase URL and API key through the UI
- **View**: See your saved credentials (API key is partially hidden)
- **Clear**: Remove saved credentials from localStorage
- **Auto-connect**: Credentials are loaded automatically on page refresh

### Using the Supabase Client

The Supabase client is initialized in `lib/supabase.ts`:

```typescript
import { getSupabaseClient } from '@/lib/supabase';

// Get client with saved credentials (or env variables)
const supabase = getSupabaseClient();

// Example: Query data
const { data, error } = await supabase
  .from('your_table')
  .select('*');
```

## Local PostgreSQL Database

A Docker-based PostgreSQL setup is included in the `database/` directory:

**Quick Start:**
```bash
cd database
./start.sh
```

**Connection Details:**
- Host: `localhost`
- Port: `5432`
- Database: `nexhacks`
- Username: `postgres`
- Password: `postgres`

**Sample Tables:**
- `sample_data` - 3 sample items for testing
- `users` - 2 sample users

See `database/README.md` for full documentation.

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm start` - Start production server
- `npm run lint` - Run ESLint

## Next Steps

1. Set up your Supabase credentials via the dashboard UI
2. Or run the local PostgreSQL database for testing
3. Create additional tables in your database
4. Build your application features
5. Add more pages in the `app/` directory
6. Create reusable components

## Learn More

- [Next.js Documentation](https://nextjs.org/docs)
- [Supabase Documentation](https://supabase.com/docs)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
