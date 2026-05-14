<script>
	import { api } from '$lib/api.js';
	import { modelStore } from '$lib/store.svelte.js';

	let datasets = $state({});
	let modelInfo = $state(null);
	let datasetName = $state('');
	let targetName = $state('');
	let maxSearchSteps = $state(20);
	let maxDepth = $state(5);
	let numEpochs = $state(40);

	let running = $state(false);
	let loading = $state(false);
	let error = $state(null);
	let success = $state(null);
	let result = $state(null);
	let evalResult = $state(null);
	let liveLogs = $state([]);
	let streamState = $state('idle');

	function suggestedTarget(name) {
		if (!name) return '';
		return name
			.toUpperCase()
			.replace(/[^A-Z0-9]+/g, '_')
			.replace(/^_+|_+$/g, '');
	}

	const datasetEntries = $derived.by(() => Object.entries(datasets));
	const selectedDatasetMeta = $derived.by(() => datasets[datasetName] ?? null);
	const trainHistory = $derived.by(() => modelInfo?.train_history ?? []);

	async function refresh() {
		loading = true;
		error = null;
		try {
			datasets = await api.listDatasets();
			modelInfo = await api.getModel(modelStore.name);

			if (!datasetName || !datasets[datasetName]) {
				datasetName = Object.keys(datasets)[0] ?? '';
			}
			if (!targetName) {
				targetName = suggestedTarget(datasetName);
			}
		} catch (e) {
			error = e.message;
		} finally {
			loading = false;
		}
	}

	function onDatasetChange(e) {
		datasetName = e.target.value;
		if (!targetName.trim()) {
			targetName = suggestedTarget(datasetName);
		}
	}

	async function runTraining() {
		error = null;
		success = null;
		result = null;
		evalResult = null;
		liveLogs = [];
		streamState = 'starting';

		if (!datasetName) {
			error = 'Choose a dataset first';
			return;
		}
		if (!targetName.trim()) {
			error = 'Target function name is required';
			return;
		}

		running = true;
		try {
			const req = {
				model_name: modelStore.name,
				dataset_name: datasetName,
				target_name: targetName.trim(),
				max_search_steps: Number(maxSearchSteps),
				max_depth: Number(maxDepth),
				num_epochs: Number(numEpochs)
			};

			let streamedResult = null;
			let streamedError = null;

			await api.trainStream(req, ({ event, data }) => {
				if (event === 'start') {
					streamState = 'running';
					liveLogs = [...liveLogs, `Started training ${data.target_name} on ${data.dataset_name}`];
					return;
				}
				if (event === 'log') {
					liveLogs = [...liveLogs, data.line ?? ''];
					return;
				}
				if (event === 'error') {
					streamedError = data.message || 'Training failed';
					liveLogs = [...liveLogs, `ERROR: ${streamedError}`];
					streamState = 'error';
					return;
				}
				if (event === 'done') {
					streamedResult = data;
					streamState = 'done';
				}
			});

			if (streamedError) {
				error = streamedError;
				return;
			}

			if (!streamedResult || !Object.prototype.hasOwnProperty.call(streamedResult, 'success')) {
				error = 'Training stream ended without a final result.';
				streamState = 'error';
				return;
			}

			result = streamedResult;

			success = result.success
				? `Training succeeded (R²=${result.r2_score})`
				: `Training finished but did not reach target fit (R²=${result.r2_score})`;

			evalResult = await api.testEval(modelStore.name, datasetName);
			await refresh();
		} catch (e) {
			error = e.message;
			streamState = 'error';
		} finally {
			running = false;
		}
	}

	$effect(() => {
		modelStore.name;
		refresh();
	});
</script>

<h1>Train</h1>
<p class="muted">
	Train the active model on a dataset to discover and register a new target function in the
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
			<span class="muted small">Vocabulary</span>
			<span class="mono">{modelInfo.vocab_size}</span>
			<span class="muted small">Functions</span>
			<span class="mono">{modelInfo.functions.length}</span>
		{/if}
	</div>
</div>

<div class="grid two-col">
	<div class="card stack">
		<h2>Training Request</h2>
		<label>
			Dataset
			<select value={datasetName} onchange={onDatasetChange} disabled={datasetEntries.length === 0}>
				<option value="" disabled selected={datasetName === ''}>Select dataset</option>
				{#each datasetEntries as [name, meta]}
					<option value={name}>{name} ({meta.num_examples} rows)</option>
				{/each}
			</select>
		</label>

		{#if selectedDatasetMeta}
			<div class="muted small dataset-note">
				{selectedDatasetMeta.description || 'No description'}
			</div>
		{/if}

		<label>
			Target Function Name
			<input type="text" bind:value={targetName} placeholder="FORCE" />
		</label>

		<div class="grid params-grid">
			<label>
				Max Search Steps
				<input type="number" min="1" max="200" bind:value={maxSearchSteps} />
			</label>
			<label>
				Max Depth
				<input type="number" min="1" max="10" bind:value={maxDepth} />
			</label>
			<label>
				Epochs
				<input type="number" min="1" max="500" bind:value={numEpochs} />
			</label>
		</div>

		<div class="row">
			<button class="primary" onclick={runTraining} disabled={running || loading || !datasetName}>
				{running ? 'Training...' : 'Train Model'}
			</button>
			<button onclick={refresh} disabled={running || loading}>Refresh model state</button>
		</div>
	</div>

	<div class="stack">
		<div class="card">
			<h2>Live Training Log</h2>
			<div class="row">
				<span class="muted small">Stream state</span>
				<span class="badge {streamState === 'error' ? 'err' : streamState === 'done' ? 'ok' : 'accent'}"
					>{streamState}</span
				>
			</div>
			<div class="log-box">
				{#if liveLogs.length === 0}
					<p class="muted small">
						{running
							? 'Waiting for first log lines from server...'
							: 'Click Train Model to stream server logs here.'}
					</p>
				{:else}
					<pre class="mono">{liveLogs.join('\n')}</pre>
				{/if}
			</div>
		</div>

		{#if result}
			<div class="card">
				<h2>Last Run</h2>
				<div class="grid metrics-grid">
					<div class="metric">
						<div class="muted small">Success</div>
						<div class={result.success ? 'badge ok' : 'badge err'}>
							{result.success ? 'yes' : 'no'}
						</div>
					</div>
					<div class="metric">
						<div class="muted small">R²</div>
						<div class="mono">{result.r2_score}</div>
					</div>
					<div class="metric">
						<div class="muted small">Elapsed</div>
						<div class="mono">{result.elapsed_s}s</div>
					</div>
					<div class="metric">
						<div class="muted small">Vocab Size</div>
						<div class="mono">{result.vocab_size}</div>
					</div>
				</div>
			</div>
		{/if}

		{#if evalResult}
			<div class="card">
				<h2>Dataset Evaluation</h2>
				<div class="row">
					<span class="badge {evalResult.accuracy >= 0.9 ? 'ok' : 'err'}">
						accuracy: {evalResult.accuracy}
					</span>
					<span class="badge accent">best R²: {evalResult.best_r2}</span>
					<span class="mono">best fn: {evalResult.best_function || 'none'}</span>
				</div>
			</div>
		{/if}
	</div>
</div>

<div class="card">
	<h2>Training History</h2>
	{#if trainHistory.length === 0}
		<p class="muted">No training runs yet for this model in the current server session.</p>
	{:else}
		<div class="table-wrap">
			<table>
				<thead>
					<tr>
						<th>Time</th>
						<th>Target</th>
						<th>Dataset</th>
						<th>Success</th>
						<th>R²</th>
						<th>Elapsed</th>
					</tr>
				</thead>
				<tbody>
					{#each trainHistory as item}
						<tr>
							<td class="mono small">{item.timestamp || '-'}</td>
							<td class="mono">{item.target || item.target_name || '-'}</td>
							<td class="mono">{item.dataset || item.dataset_name || '-'}</td>
							<td>
								<span class={item.success ? 'badge ok' : 'badge err'}>
									{item.success ? 'yes' : 'no'}
								</span>
							</td>
							<td class="mono">{item.r2_score ?? '-'}</td>
							<td class="mono">{item.elapsed_s ?? '-'}s</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
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

	.params-grid {
		grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
	}

	.dataset-note {
		padding: 0.5rem 0.6rem;
		border: 1px dashed var(--border);
		border-radius: 6px;
		background: var(--bg);
	}

	.metrics-grid {
		grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
	}

	.metric {
		padding: 0.55rem 0.65rem;
		border: 1px solid var(--border);
		border-radius: 6px;
		background: var(--bg);
	}

	.table-wrap {
		overflow: auto;
		max-height: 360px;
		border: 1px solid var(--border);
		border-radius: 6px;
	}

	.log-box {
		margin-top: 0.65rem;
		max-height: 420px;
		overflow: auto;
		padding: 0.75rem;
		border: 1px solid var(--border);
		border-radius: 6px;
		background: var(--bg);
	}

	.log-box pre {
		margin: 0;
		white-space: pre-wrap;
		word-break: break-word;
		font-size: 0.82rem;
		line-height: 1.35;
	}
</style>
