<script>
	import { page } from '$app/state';
	import { modelStore } from '$lib/store.svelte.js';

	const links = [
		{ href: '/', label: 'Home' },
		{ href: '/equations', label: 'Equations' },
		{ href: '/datasets', label: 'Datasets' },
		{ href: '/graphs', label: 'Graphs' },
		{ href: '/train', label: 'Train' },
		{ href: '/predict', label: 'Predict' }
	];

	function isActive(href) {
		if (href === '/') return page.url.pathname === '/';
		return page.url.pathname.startsWith(href);
	}

	function onModelChange(e) {
		modelStore.set(e.target.value);
	}
</script>

<nav class="navbar">
	<div class="brand">
		<span class="logo">∑</span>
		<span class="brand-text">NSRR</span>
	</div>

	<ul class="links">
		{#each links as link}
			<li>
				<a href={link.href} class:active={isActive(link.href)}>{link.label}</a>
			</li>
		{/each}
	</ul>

	<div class="model-picker">
		<label for="model-name">Model</label>
		<input
			id="model-name"
			type="text"
			value={modelStore.name}
			oninput={onModelChange}
			placeholder="default"
		/>
	</div>
</nav>

<style>
	.navbar {
		display: flex;
		align-items: center;
		gap: 2rem;
		padding: 0.75rem 1.5rem;
		background: var(--bg-elev);
		border-bottom: 1px solid var(--border);
		position: sticky;
		top: 0;
		z-index: 10;
	}

	.brand {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		font-weight: 700;
	}

	.logo {
		font-size: 1.6rem;
		color: var(--accent);
		line-height: 1;
	}

	.brand-text {
		font-size: 1.1rem;
		letter-spacing: 0.04em;
	}

	.links {
		list-style: none;
		display: flex;
		gap: 0.25rem;
		margin: 0;
		padding: 0;
		flex: 1;
	}

	.links a {
		display: block;
		padding: 0.4rem 0.85rem;
		border-radius: 6px;
		text-decoration: none;
		color: var(--text-dim);
		font-size: 0.95rem;
	}

	.links a:hover {
		background: var(--bg-hover);
		color: var(--text);
	}

	.links a.active {
		background: var(--accent-dim);
		color: var(--accent);
	}

	.model-picker {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}

	.model-picker label {
		font-size: 0.85rem;
		color: var(--text-dim);
	}

	.model-picker input {
		width: 120px;
	}
</style>
