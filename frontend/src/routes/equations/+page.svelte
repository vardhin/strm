<script>
	import { api } from '$lib/api.js';
	import { modelStore } from '$lib/store.svelte.js';
	import { renderFunction, PRIMITIVE_DESCRIPTIONS } from '$lib/math_render.js';
	import { onMount } from 'svelte';

	let model = $state(null);
	let error = $state(null);
	let loading = $state(false);
	let filter = $state('all'); // all | primitive | learned
	let query = $state('');
	let expanded = $state({});
	let selected = $state(null);

	async function load() {
		loading = true;
		error = null;
		try {
			model = await api.getModel(modelStore.name);
		} catch (e) {
			error = e.message;
		} finally {
			loading = false;
		}
	}

	$effect(() => {
		// Reload when model name changes
		modelStore.name;
		load();
	});

	const filtered = $derived.by(() => {
		if (!model) return [];
		let list = model.functions;
		if (filter === 'primitive') list = list.filter((f) => f.layer === 0);
		else if (filter === 'learned') list = list.filter((f) => f.layer > 0);
		if (query.trim()) {
			const q = query.toLowerCase();
			list = list.filter((f) => f.name.toLowerCase().includes(q));
		}
		return list;
	});

	function toggle(id) {
		expanded[id] = !expanded[id];
	}

	function select(fn) {
		selected = selected && selected.id === fn.id ? null : fn;
	}

	function prefillFromSelected() {
		if (!selected || selected.layer === 0 || !selected.composition) return;
		creatingOpen = true;
		editError = null;
		editSuccess = null;
		editName = `${selected.name}_v2`;
		editArity = selected.arity;
		editCompText = JSON.stringify(
			selected.composition.map((step) => [step.func_id, step.args]),
			null,
			2
		);
	}

	function rendered(fn) {
		if (!model) return '';
		try {
			return renderFunction(fn, model.functions);
		} catch (e) {
			return `(render error: ${e.message})`;
		}
	}

	// Edit mode for learned fn constants (the only editable bit we can round-trip
	// without re-registering the composition — since the backend doesn't currently
	// expose an UPDATE endpoint, editing rebuilds via re-registering under a new name).
	let editName = $state('');
	let editArity = $state(2);
	let editCompText = $state('');
	let editError = $state(null);
	let editSuccess = $state(null);
	let creatingOpen = $state(false);

	async function createComposed() {
		editError = null;
		editSuccess = null;
		let composition;
		try {
			composition = JSON.parse(editCompText);
			if (!Array.isArray(composition)) throw new Error('Composition must be a JSON array');
		} catch (e) {
			editError = `Invalid JSON: ${e.message}`;
			return;
		}
		try {
			const res = await api.registerFunction(
				modelStore.name,
				editName.trim(),
				Number(editArity),
				composition
			);
			editSuccess = `Registered '${editName}' with id ${res.func_id}`;
			editName = '';
			editCompText = '';
			await load();
		} catch (e) {
			editError = e.message;
		}
	}

	onMount(() => {
		load();
		const timer = setInterval(load, 5000);
		return () => clearInterval(timer);
	});
</script>

<h1>Equations</h1>
<p class="muted">
	Registered functions for model <span class="mono">{modelStore.name}</span>. Primitives (layer 0)
	are built-in; learned functions (layer ≥ 1) are discovered from training.
</p>

<div class="card">
	<div class="row">
		<label>
			Filter:
			<select bind:value={filter}>
				<option value="all">All</option>
				<option value="primitive">Primitives</option>
				<option value="learned">Learned</option>
			</select>
		</label>
		<label>
			Search:
			<input type="text" bind:value={query} placeholder="name…" />
		</label>
		<button onclick={load} disabled={loading}>{loading ? 'Loading…' : 'Reload'}</button>
		<span class="muted small">
			{model?.functions.length ?? 0} total · vocab {model?.vocab_size ?? '?'}
		</span>
	</div>
</div>

{#if error}
	<div class="error">{error}</div>
{/if}

{#if model}
	<div class="card">
		<table>
			<thead>
				<tr>
					<th class="col-id">ID</th>
					<th class="col-name">Name</th>
					<th class="col-small">Arity</th>
					<th class="col-small">Layer</th>
					<th>Expression</th>
				</tr>
			</thead>
			<tbody>
				{#each filtered as fn}
					<tr class:selected={selected?.id === fn.id} onclick={() => select(fn)}>
						<td class="mono">{fn.id}</td>
						<td class="mono">{fn.name}</td>
						<td>{fn.arity}</td>
						<td>
							{#if fn.layer === 0}
								<span class="badge">primitive</span>
							{:else}
								<span class="badge accent">L{fn.layer}</span>
							{/if}
						</td>
						<td class="mono expr">
							{#if fn.layer === 0}
								<span class="muted">{PRIMITIVE_DESCRIPTIONS[fn.name] ?? fn.name}</span>
							{:else}
								{rendered(fn)}
							{/if}
						</td>
					</tr>
				{/each}
				{#if filtered.length === 0}
					<tr>
						<td colspan="5" class="muted">No matches.</td>
					</tr>
				{/if}
			</tbody>
		</table>
	</div>

	{#if selected}
		<div class="card">
			<h2>
				{selected.name}
				{#if selected.layer === 0}
					<span class="badge">primitive</span>
				{:else}
					<span class="badge accent">learned · L{selected.layer}</span>
				{/if}
			</h2>

			<p class="mono big">
				{selected.name}({Array.from({ length: selected.arity }, (_, i) => `x${i}`).join(', ')}) = {rendered(
					selected
				)}
			</p>

			{#if selected.layer === 0}
				<p class="muted">{PRIMITIVE_DESCRIPTIONS[selected.name] ?? 'Built-in primitive.'}</p>
			{:else if selected.composition}
				<div class="row">
					<button onclick={prefillFromSelected}>Edit Composition As New Function</button>
					<span class="muted small">
						Registry entries are immutable, so edits are saved as a new function name.
					</span>
				</div>

				<h3>Composition steps</h3>
				<table class="steps">
					<thead>
						<tr>
							<th>Step</th>
							<th>Function</th>
							<th>Arg indices</th>
						</tr>
					</thead>
					<tbody>
						{#each selected.composition as step, i}
							<tr>
								<td class="mono">{selected.arity + i}</td>
								<td class="mono">
									<span class="badge">{step.func_id}</span>
									{step.func_name}
								</td>
								<td class="mono">[{step.args.join(', ')}]</td>
							</tr>
						{/each}
					</tbody>
				</table>
				<p class="muted small">
					Arg indices reference the "available values" list that starts with inputs x₀..x{selected.arity -
						1} and grows by 1 with each step's result.
				</p>

				{#if selected.constants}
					<p>
						<strong>Constants:</strong>
						<span class="mono">{JSON.stringify(selected.constants)}</span>
						<span class="muted">({selected.const_mode})</span>
					</p>
				{/if}
			{/if}
		</div>
	{/if}

	<div class="card">
		<button onclick={() => (creatingOpen = !creatingOpen)}>
			{creatingOpen ? '▼' : '▶'} Register a new composed function
		</button>
		{#if creatingOpen}
			<div class="stack form">
				<p class="muted small">
					Create a new learned function by specifying a composition list: an array of
					<span class="mono">[func_id, [arg_indices]]</span> pairs. Arg indices refer to the inputs
					first (0..arity-1), then each step's result (appended).
				</p>
				<label>
					Name
					<input type="text" bind:value={editName} placeholder="MY_FN" />
				</label>
				<label>
					Arity
					<input type="number" min="0" max="8" bind:value={editArity} />
				</label>
				<label>
					Composition (JSON)
					<textarea
						bind:value={editCompText}
						placeholder={'[[0, [0, 1]], [3, [0, 2]]]  // ex: OR(x0,x1) then ADD(x0, step0)'}
					></textarea>
				</label>
				<div class="row">
					<button class="primary" onclick={createComposed} disabled={!editName}>Register</button>
				</div>
				{#if editError}<div class="error">{editError}</div>{/if}
				{#if editSuccess}<div class="success">{editSuccess}</div>{/if}
			</div>
		{/if}
	</div>
{/if}

<style>
	tr.selected td {
		background: var(--accent-dim);
	}

	tr {
		cursor: pointer;
	}

	.expr {
		color: var(--text);
		font-size: 0.95rem;
	}

	.big {
		font-size: 1.1rem;
		padding: 0.75rem;
		background: var(--bg);
		border-radius: 6px;
		border: 1px solid var(--border);
	}

	.steps th,
	.steps td {
		font-size: 0.9rem;
	}

	.form label {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
		font-size: 0.9rem;
		color: var(--text-dim);
	}

	.form input[type='text'],
	.form input[type='number'] {
		max-width: 320px;
	}

	.col-id {
		width: 3rem;
	}

	.col-name {
		width: 14rem;
	}

	.col-small {
		width: 4rem;
	}
</style>
