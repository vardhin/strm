<script>
	import { api } from '$lib/api.js';
	import { modelStore } from '$lib/store.svelte.js';
	import { onMount } from 'svelte';

	let health = $state(null);
	let models = $state(null);
	let error = $state(null);

	async function load() {
		error = null;
		try {
			health = await api.health();
			models = await api.listModels();
		} catch (e) {
			error = e.message;
		}
	}

	onMount(load);
</script>

<h1>NSRR Dashboard</h1>
<p class="muted">
	Neuro-Symbolic Recursive Regression — discover equations from data using a tiny recursive
	reasoning model and a symbolic function registry.
</p>

{#if error}
	<div class="error">Can't reach backend at localhost:8000 — {error}</div>
{/if}

{#if health}
	<div class="card">
		<div class="row">
			<span class="badge ok">{health.status}</span>
			<span class="muted small">Loaded models:</span>
			<span class="mono">{health.loaded_models.join(', ') || 'none'}</span>
			<span class="muted small">Datasets:</span>
			<span class="mono">{health.datasets.length}</span>
			<span class="muted small">Experiments run:</span>
			<span class="mono">{health.experiments_run}</span>
		</div>
	</div>
{/if}

<h2>Workflow</h2>
<div class="grid tiles">
	<a href="/equations" class="tile">
		<div class="tile-icon">ƒ</div>
		<div class="tile-body">
			<h3>Equations</h3>
			<p class="muted small">Browse primitives and learned functions. Edit compositions.</p>
		</div>
	</a>
	<a href="/datasets" class="tile">
		<div class="tile-icon">⊟</div>
		<div class="tile-body">
			<h3>Datasets</h3>
			<p class="muted small">Create, import, edit and export datasets (CSV).</p>
		</div>
	</a>
	<a href="/graphs" class="tile">
		<div class="tile-icon">◢</div>
		<div class="tile-body">
			<h3>Graphs</h3>
			<p class="muted small">Visualise dataset points for any input column vs output.</p>
		</div>
	</a>
	<a href="/train" class="tile">
		<div class="tile-icon">◉</div>
		<div class="tile-body">
			<h3>Train</h3>
			<p class="muted small">Teach the model a new target function from a dataset.</p>
		</div>
	</a>
	<a href="/predict" class="tile">
		<div class="tile-icon">▷</div>
		<div class="tile-body">
			<h3>Predict</h3>
			<p class="muted small">Run any learned function on fresh input values.</p>
		</div>
	</a>
</div>

{#if models}
	<h2>Loaded Models</h2>
	<div class="card">
		<table>
			<thead>
				<tr>
					<th>Name</th>
					<th>Functions</th>
					<th>Vocab</th>
					<th>Trainings</th>
				</tr>
			</thead>
			<tbody>
				{#each Object.entries(models) as [name, info]}
					<tr>
						<td>
							<button
								class="linklike"
								onclick={() => modelStore.set(name)}
								class:selected={modelStore.name === name}>{name}</button
							>
						</td>
						<td>{info.num_functions}</td>
						<td>{info.vocab_size}</td>
						<td>{info.train_history_count}</td>
					</tr>
				{/each}
			</tbody>
		</table>
		<p class="muted small">Click a name to select it as the active model.</p>
	</div>
{/if}

<style>
	.tiles {
		grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
	}

	.tile {
		display: flex;
		gap: 0.9rem;
		padding: 1rem;
		background: var(--bg-elev);
		border: 1px solid var(--border);
		border-radius: 8px;
		text-decoration: none;
		color: inherit;
		transition: border 0.15s, transform 0.15s;
	}

	.tile:hover {
		border-color: var(--accent);
		transform: translateY(-2px);
	}

	.tile-icon {
		font-size: 1.8rem;
		color: var(--accent);
		line-height: 1;
	}

	.tile-body h3 {
		margin: 0 0 0.25rem 0;
	}

	.tile-body p {
		margin: 0;
	}

	button.linklike {
		background: transparent;
		border: none;
		padding: 0;
		color: var(--accent);
		cursor: pointer;
	}

	button.linklike.selected {
		font-weight: 700;
	}
</style>
