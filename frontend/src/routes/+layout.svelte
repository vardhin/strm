<script>
	import favicon from '$lib/assets/favicon.svg';
	import Navbar from '$lib/components/Navbar.svelte';
	import { modelStore } from '$lib/store.svelte.js';
	import { onMount } from 'svelte';

	let { children } = $props();

	onMount(() => {
		modelStore.load();
	});
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
	<title>NSSR</title>
</svelte:head>

<div class="app">
	<Navbar />
	<main class="content">
		{@render children()}
	</main>
</div>

<style>
	:global(:root) {
		--bg: #0e1117;
		--bg-elev: #161b22;
		--bg-hover: #1f2630;
		--border: #2a3038;
		--text: #e6edf3;
		--text-dim: #8b949e;
		--accent: #58a6ff;
		--accent-dim: rgba(88, 166, 255, 0.14);
		--success: #3fb950;
		--success-dim: rgba(63, 185, 80, 0.14);
		--danger: #f85149;
		--danger-dim: rgba(248, 81, 73, 0.14);
		--warn: #d29922;
		--mono: ui-monospace, 'JetBrains Mono', 'Fira Code', Menlo, Consolas, monospace;
		--sans: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
	}

	:global(*),
	:global(*::before),
	:global(*::after) {
		box-sizing: border-box;
	}

	:global(html, body) {
		margin: 0;
		padding: 0;
		background: var(--bg);
		color: var(--text);
		font-family: var(--sans);
		font-size: 15px;
		line-height: 1.5;
		min-height: 100vh;
	}

	:global(.app-shell) {
		display: contents;
	}

	:global(h1, h2, h3, h4) {
		margin: 0 0 0.75rem 0;
		font-weight: 600;
		line-height: 1.3;
	}

	:global(h1) {
		font-size: 1.75rem;
	}

	:global(h2) {
		font-size: 1.35rem;
	}

	:global(h3) {
		font-size: 1.1rem;
	}

	:global(p) {
		margin: 0 0 0.75rem 0;
	}

	:global(a) {
		color: var(--accent);
	}

	:global(button) {
		font-family: inherit;
		font-size: 0.95rem;
		padding: 0.45rem 0.9rem;
		background: var(--bg-elev);
		color: var(--text);
		border: 1px solid var(--border);
		border-radius: 6px;
		cursor: pointer;
		transition: background 0.15s;
	}

	:global(button:hover:not(:disabled)) {
		background: var(--bg-hover);
	}

	:global(button:disabled) {
		opacity: 0.5;
		cursor: not-allowed;
	}

	:global(button.primary) {
		background: var(--accent);
		color: #0b1220;
		border-color: var(--accent);
		font-weight: 600;
	}

	:global(button.primary:hover:not(:disabled)) {
		filter: brightness(1.1);
	}

	:global(button.danger) {
		background: transparent;
		color: var(--danger);
		border-color: var(--danger);
	}

	:global(button.danger:hover:not(:disabled)) {
		background: var(--danger-dim);
	}

	:global(input, select, textarea) {
		font-family: inherit;
		font-size: 0.95rem;
		padding: 0.4rem 0.6rem;
		background: var(--bg);
		color: var(--text);
		border: 1px solid var(--border);
		border-radius: 6px;
	}

	:global(input:focus, select:focus, textarea:focus) {
		outline: none;
		border-color: var(--accent);
		box-shadow: 0 0 0 2px var(--accent-dim);
	}

	:global(textarea) {
		width: 100%;
		min-height: 100px;
		font-family: var(--mono);
		resize: vertical;
	}

	:global(code, .mono) {
		font-family: var(--mono);
		font-size: 0.9em;
	}

	:global(.card) {
		background: var(--bg-elev);
		border: 1px solid var(--border);
		border-radius: 8px;
		padding: 1.1rem 1.25rem;
		margin-bottom: 1rem;
	}

	:global(.row) {
		display: flex;
		gap: 0.75rem;
		align-items: center;
		flex-wrap: wrap;
	}

	:global(.stack) {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
	}

	:global(.grid) {
		display: grid;
		gap: 1rem;
	}

	:global(.muted) {
		color: var(--text-dim);
	}

	:global(.small) {
		font-size: 0.85rem;
	}

	:global(.badge) {
		display: inline-block;
		padding: 0.1rem 0.5rem;
		font-size: 0.75rem;
		border-radius: 4px;
		background: var(--bg-hover);
		color: var(--text-dim);
		border: 1px solid var(--border);
		font-family: var(--mono);
	}

	:global(.badge.ok) {
		background: var(--success-dim);
		color: var(--success);
		border-color: transparent;
	}

	:global(.badge.err) {
		background: var(--danger-dim);
		color: var(--danger);
		border-color: transparent;
	}

	:global(.badge.accent) {
		background: var(--accent-dim);
		color: var(--accent);
		border-color: transparent;
	}

	:global(table) {
		width: 100%;
		border-collapse: collapse;
	}

	:global(th, td) {
		text-align: left;
		padding: 0.5rem 0.75rem;
		border-bottom: 1px solid var(--border);
	}

	:global(th) {
		color: var(--text-dim);
		font-weight: 500;
		font-size: 0.85rem;
		text-transform: uppercase;
		letter-spacing: 0.03em;
	}

	:global(tr:hover td) {
		background: var(--bg-hover);
	}

	:global(.error) {
		color: var(--danger);
		padding: 0.5rem 0.75rem;
		border-radius: 6px;
		background: var(--danger-dim);
		border: 1px solid var(--danger);
	}

	:global(.success) {
		color: var(--success);
		padding: 0.5rem 0.75rem;
		border-radius: 6px;
		background: var(--success-dim);
		border: 1px solid var(--success);
	}

	.app {
		min-height: 100vh;
	}

	.content {
		max-width: 1200px;
		margin: 0 auto;
		padding: 1.5rem;
	}
</style>
