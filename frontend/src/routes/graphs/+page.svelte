<script>
	import { api } from '$lib/api.js';
	import ScatterPlot from '$lib/components/ScatterPlot.svelte';
	import { onMount } from 'svelte';

	let datasets = $state({});
	let selectedName = $state('');
	let selectedDataset = $state(null);
	let xIndex = $state(0);
	let yMode = $state('output');
	let loading = $state(false);
	let error = $state(null);

	async function loadDatasetList() {
		datasets = await api.listDatasets();
	}

	async function loadDataset(name) {
		if (!name) {
			selectedDataset = null;
			selectedName = '';
			return;
		}
		loading = true;
		error = null;
		try {
			const ds = await api.getDataset(name);
			selectedName = name;
			selectedDataset = ds;
			xIndex = 0;
			yMode = 'output';
		} catch (e) {
			error = e.message;
		} finally {
			loading = false;
		}
	}

	async function refresh() {
		error = null;
		try {
			await loadDatasetList();
			if (selectedName && datasets[selectedName]) {
				await loadDataset(selectedName);
				return;
			}
			const first = Object.keys(datasets)[0] ?? '';
			if (first) await loadDataset(first);
			else selectedDataset = null;
		} catch (e) {
			error = e.message;
		}
	}

	const inputCount = $derived.by(() => {
		if (!selectedDataset || selectedDataset.examples.length === 0) return 0;
		let max = 0;
		for (const [inp] of selectedDataset.examples) {
			if (inp.length > max) max = inp.length;
		}
		return max;
	});

	const yOptions = $derived.by(() => {
		const opts = [{ value: 'output', label: 'output' }];
		for (let i = 0; i < inputCount; i++) {
			opts.push({ value: `input_${i}`, label: `input_${i}` });
		}
		return opts;
	});

	const points = $derived.by(() => {
		if (!selectedDataset) return [];
		const rows = [];
		for (const [inp, out] of selectedDataset.examples) {
			const x = Number(inp[xIndex]);
			let y;
			if (yMode === 'output') {
				y = Number(out);
			} else {
				const idx = Number(yMode.replace('input_', ''));
				y = Number(inp[idx]);
			}
			if (Number.isFinite(x) && Number.isFinite(y)) {
				rows.push([x, y]);
			}
		}
		return rows;
	});

	const summary = $derived.by(() => {
		if (points.length === 0) return null;
		let xMin = Infinity;
		let xMax = -Infinity;
		let yMin = Infinity;
		let yMax = -Infinity;
		let xSum = 0;
		let ySum = 0;

		for (const [x, y] of points) {
			if (x < xMin) xMin = x;
			if (x > xMax) xMax = x;
			if (y < yMin) yMin = y;
			if (y > yMax) yMax = y;
			xSum += x;
			ySum += y;
		}

		return {
			xMin,
			xMax,
			yMin,
			yMax,
			xMean: xSum / points.length,
			yMean: ySum / points.length
		};
	});

	const previewRows = $derived.by(() => {
		if (!selectedDataset) return [];
		return selectedDataset.examples.slice(0, 50);
	});

	function fmt(n) {
		if (!Number.isFinite(n)) return '-';
		if (Number.isInteger(n)) return String(n);
		const abs = Math.abs(n);
		if (abs >= 1000 || (abs < 0.01 && abs > 0)) return n.toExponential(2);
		return n.toFixed(4).replace(/\.?0+$/, '');
	}

	onMount(refresh);
</script>

<h1>Graphs</h1>
<p class="muted">
	Plot dataset columns to inspect trends and outliers before training. Choose any input column for
	x-axis and compare it against output or another input column.
</p>

{#if error}
	<div class="error">{error}</div>
{/if}

<div class="card controls">
	<label>
		Dataset
		<select
			value={selectedName}
			onchange={(e) => loadDataset(e.target.value)}
			disabled={Object.keys(datasets).length === 0}
		>
			<option value="" disabled selected={selectedName === ''}>Select dataset</option>
			{#each Object.keys(datasets) as name}
				<option value={name}>{name}</option>
			{/each}
		</select>
	</label>

	<label>
		X axis
		<select bind:value={xIndex}>
			{#each Array(inputCount) as _, i}
				<option value={i}>input_{i}</option>
			{/each}
		</select>
	</label>

	<label>
		Y axis
		<select bind:value={yMode}>
			{#each yOptions as opt}
				<option value={opt.value}>{opt.label}</option>
			{/each}
		</select>
	</label>

	<button onclick={refresh} disabled={loading}>
		{loading ? 'Loading...' : 'Reload'}
	</button>
</div>

{#if selectedDataset}
	<div class="card">
		<div class="row">
			<h2 class="grow-title mono">{selectedDataset.name}</h2>
			<span class="badge">{selectedDataset.examples.length} rows</span>
			<span class="badge accent">{inputCount} inputs</span>
		</div>
		{#if selectedDataset.description}
			<p class="muted">{selectedDataset.description}</p>
		{/if}

		{#if points.length > 0}
			<ScatterPlot points={points} xLabel={`input_${xIndex}`} yLabel={yMode} />
		{:else}
			<p class="muted">No plottable rows for this axis selection.</p>
		{/if}
	</div>

	{#if summary}
		<div class="card">
			<h3>Axis Stats</h3>
			<div class="grid stats-grid">
				<div class="stat-box">
					<div class="muted small">X min</div>
					<div class="mono">{fmt(summary.xMin)}</div>
				</div>
				<div class="stat-box">
					<div class="muted small">X max</div>
					<div class="mono">{fmt(summary.xMax)}</div>
				</div>
				<div class="stat-box">
					<div class="muted small">X mean</div>
					<div class="mono">{fmt(summary.xMean)}</div>
				</div>
				<div class="stat-box">
					<div class="muted small">Y min</div>
					<div class="mono">{fmt(summary.yMin)}</div>
				</div>
				<div class="stat-box">
					<div class="muted small">Y max</div>
					<div class="mono">{fmt(summary.yMax)}</div>
				</div>
				<div class="stat-box">
					<div class="muted small">Y mean</div>
					<div class="mono">{fmt(summary.yMean)}</div>
				</div>
			</div>
		</div>
	{/if}

	<div class="card">
		<h3>Preview (first 50 rows)</h3>
		<div class="table-wrap">
			<table>
				<thead>
					<tr>
						<th>#</th>
						{#each Array(inputCount) as _, i}
							<th>input_{i}</th>
						{/each}
						<th>output</th>
					</tr>
				</thead>
				<tbody>
					{#each previewRows as [inp, out], i}
						<tr>
							<td class="mono muted">{i}</td>
							{#each Array(inputCount) as _, j}
								<td class="mono">{fmt(Number(inp[j]))}</td>
							{/each}
							<td class="mono">{fmt(Number(out))}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	</div>
{:else}
	<div class="card">
		<p class="muted">No datasets available yet. Create one in the Datasets page first.</p>
	</div>
{/if}

<style>
	.controls {
		display: flex;
		flex-wrap: wrap;
		gap: 0.8rem;
		align-items: end;
	}

	.controls label {
		display: flex;
		flex-direction: column;
		gap: 0.2rem;
		font-size: 0.9rem;
		color: var(--text-dim);
	}

	.grow-title {
		flex: 1;
		margin: 0;
	}

	.stats-grid {
		grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
	}

	.stat-box {
		border: 1px solid var(--border);
		border-radius: 6px;
		padding: 0.55rem 0.7rem;
		background: var(--bg);
	}

	.table-wrap {
		overflow: auto;
		max-height: 420px;
		border: 1px solid var(--border);
		border-radius: 6px;
	}
</style>
