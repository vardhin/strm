<script>
	import { api } from '$lib/api.js';
	import { modelStore } from '$lib/store.svelte.js';
	import { renderFunction } from '$lib/math_render.js';

	let modelInfo = $state(null);
	let includePrimitives = $state(false);
	let selectedId = $state('');
	let inputValues = $state([]);
	let loading = $state(false);
	let running = $state(false);
	let error = $state(null);
	let success = $state(null);
	let result = $state(null);
	let neuralHint = $state(null);
	let lastFnKey = $state('');

	const selectableFunctions = $derived.by(() => {
		if (!modelInfo) return [];
		let list = modelInfo.functions;
		if (!includePrimitives) list = list.filter((f) => f.layer > 0);
		return list;
	});

	const selectedFn = $derived.by(() => {
		const id = Number(selectedId);
		if (!Number.isInteger(id)) return null;
		return selectableFunctions.find((f) => f.id === id) ?? null;
	});

	$effect(() => {
		const all = selectableFunctions;
		if (all.length === 0) {
			selectedId = '';
			return;
		}
		const current = Number(selectedId);
		const exists = all.some((f) => f.id === current);
		if (!exists) selectedId = String(all[0].id);
	});

	$effect(() => {
		const fn = selectedFn;
		const key = fn ? `${fn.id}:${fn.arity}` : '';
		if (key === lastFnKey) return;

		lastFnKey = key;
		if (!fn) {
			inputValues = [];
			return;
		}

		const next = [];
		for (let i = 0; i < fn.arity; i++) {
			next.push(inputValues[i] ?? '0');
		}
		inputValues = next;
	});

	async function refresh() {
		loading = true;
		error = null;
		try {
			modelInfo = await api.getModel(modelStore.name);
		} catch (e) {
			error = e.message;
		} finally {
			loading = false;
		}
	}

	function updateInput(i, value) {
		const next = inputValues.slice();
		next[i] = value;
		inputValues = next;
	}

	function currentInputs() {
		const nums = [];
		for (let i = 0; i < inputValues.length; i++) {
			const n = Number(inputValues[i]);
			if (!Number.isFinite(n)) {
				throw new Error(`Input ${i} is not a valid number`);
			}
			nums.push(n);
		}
		return nums;
	}

	async function runPrediction() {
		error = null;
		success = null;
		result = null;
		neuralHint = null;

		if (!selectedFn) {
			error = 'Select a function first';
			return;
		}

		running = true;
		try {
			const inputs = currentInputs();
			result = await api.executeFunction(modelStore.name, selectedFn.id, inputs);
			success = `Computed ${selectedFn.name}(${inputs.join(', ')})`;

			// Optional: inspect what the neural model proposes for these inputs.
			neuralHint = await api.testPredict(modelStore.name, inputs);
		} catch (e) {
			error = e.message;
		} finally {
			running = false;
		}
	}

	$effect(() => {
		modelStore.name;
		refresh();
	});
</script>

<h1>Predict</h1>
<p class="muted">
	Select a learned function, provide input values, and compute the output from the symbolic
	registry.
</p>

{#if error}
	<div class="error">{error}</div>
{/if}
{#if success}
	<div class="success">{success}</div>
{/if}

<div class="card">
	<div class="row">
		<span class="muted small">Active model</span>
		<span class="badge accent mono">{modelStore.name}</span>
		{#if modelInfo}
			<span class="muted small">Functions available</span>
			<span class="mono">{selectableFunctions.length}</span>
		{/if}
		<button onclick={refresh} disabled={loading || running}>{loading ? 'Loading...' : 'Refresh'}</button>
	</div>
</div>

<div class="grid two-col">
	<div class="card stack">
		<h2>Function Call</h2>

		<label class="row checkbox-row">
			<input type="checkbox" bind:checked={includePrimitives} />
			<span>Include primitive functions</span>
		</label>

		<label>
			Function
			<select bind:value={selectedId} disabled={selectableFunctions.length === 0}>
				{#each selectableFunctions as fn}
					<option value={String(fn.id)}>{fn.name} (id={fn.id}, arity={fn.arity})</option>
				{/each}
			</select>
		</label>

		{#if selectedFn}
			<div class="formula mono">
				{selectedFn.name}({Array.from({ length: selectedFn.arity }, (_, i) => `x${i}`).join(', ')}) =
				{selectedFn.layer > 0 ? renderFunction(selectedFn, modelInfo.functions) : selectedFn.name}
			</div>

			<div class="grid inputs-grid">
				{#each Array(selectedFn.arity) as _, i}
					<label>
						x{i}
						<input
							type="number"
							step="any"
							value={inputValues[i] ?? '0'}
							oninput={(e) => updateInput(i, e.target.value)}
						/>
					</label>
				{/each}
			</div>

			<div class="row">
				<button class="primary" onclick={runPrediction} disabled={running}>
					{running ? 'Running...' : 'Compute Output'}
				</button>
			</div>
		{:else}
			<p class="muted">No functions available. Train a model first or enable primitive functions.</p>
		{/if}
	</div>

	<div class="stack">
		{#if result}
			<div class="card">
				<h2>Result</h2>
				<div class="result-box mono">{result.result}</div>
				<p class="muted small">func_id: {result.func_id} | inputs: [{result.inputs.join(', ')}]</p>
			</div>
		{/if}

		{#if neuralHint}
			<div class="card">
				<h2>Neural Suggestion</h2>
				<p class="muted small">
					Top neural predictions for the same inputs (useful for debugging search behavior).
				</p>
				<div class="row">
					<span class="badge">halt_prob: {neuralHint.predictions.halt_prob}</span>
				</div>

				<h3>Primary</h3>
				<ul class="pred-list">
					{#each neuralHint.predictions.primary as item}
						<li class="mono">#{item.id} {item.name} ({item.score})</li>
					{/each}
				</ul>

				<h3>Composition Type</h3>
				<ul class="pred-list">
					{#each Object.entries(neuralHint.predictions.composition) as [name, score]}
						<li class="mono">{name}: {score}</li>
					{/each}
				</ul>
			</div>
		{/if}
	</div>
</div>

<style>
	.two-col {
		grid-template-columns: 1fr 1fr;
	}

	@media (max-width: 980px) {
		.two-col {
			grid-template-columns: 1fr;
		}
	}

	label {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
		font-size: 0.9rem;
		color: var(--text-dim);
	}

	.checkbox-row {
		flex-direction: row;
		align-items: center;
		gap: 0.6rem;
		color: var(--text);
	}

	.checkbox-row input {
		width: auto;
	}

	.inputs-grid {
		grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
	}

	.formula {
		padding: 0.7rem;
		border: 1px solid var(--border);
		border-radius: 6px;
		background: var(--bg);
		line-height: 1.5;
	}

	.result-box {
		font-size: 1.3rem;
		padding: 0.8rem 0.9rem;
		border: 1px solid var(--accent);
		border-radius: 6px;
		background: var(--accent-dim);
	}

	.pred-list {
		margin: 0;
		padding-left: 1rem;
		display: grid;
		gap: 0.25rem;
	}
</style>
