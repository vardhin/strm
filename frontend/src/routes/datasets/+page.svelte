<script>
	import { api } from '$lib/api.js';
	import { parseCSV, rowsToExamples, examplesToCSV } from '$lib/csv.js';
	import DatasetEditor from '$lib/components/DatasetEditor.svelte';
	import * as XLSX from 'xlsx';
	import { onMount } from 'svelte';

	let datasets = $state({});
	let selected = $state(null); // name
	let currentExamples = $state([]);
	let currentDescription = $state('');
	let numInputs = $state(2);
	let loading = $state(false);
	let error = $state(null);
	let success = $state(null);

	// Create form
	let newName = $state('');
	let newDesc = $state('');
	let createOpen = $state(false);
	let importOpen = $state(false);

	async function loadList() {
		try {
			datasets = await api.listDatasets();
		} catch (e) {
			error = e.message;
		}
	}

	async function selectDataset(name) {
		loading = true;
		error = null;
		success = null;
		try {
			const ds = await api.getDataset(name);
			selected = name;
			currentDescription = ds.description;
			currentExamples = ds.examples.map(([inp, out]) => [inp, out]);
			numInputs =
				currentExamples.length > 0 ? Math.max(1, currentExamples[0][0].length) : 2;
		} catch (e) {
			error = e.message;
		} finally {
			loading = false;
		}
	}

	async function saveCurrent() {
		if (!selected) return;
		error = null;
		success = null;
		try {
			await api.createDataset(selected, currentDescription, currentExamples);
			success = `Saved "${selected}" (${currentExamples.length} rows)`;
			await loadList();
		} catch (e) {
			error = e.message;
		}
	}

	async function deleteSelected() {
		if (!selected) return;
		if (!confirm(`Delete dataset "${selected}"?`)) return;
		try {
			await api.deleteDataset(selected);
			selected = null;
			currentExamples = [];
			await loadList();
		} catch (e) {
			error = e.message;
		}
	}

	async function createNew() {
		if (!newName.trim()) {
			error = 'Dataset name required';
			return;
		}
		try {
			await api.createDataset(newName.trim(), newDesc, []);
			await loadList();
			await selectDataset(newName.trim());
			newName = '';
			newDesc = '';
			createOpen = false;
		} catch (e) {
			error = e.message;
		}
	}

	async function parseTabularFile(file) {
		const name = file.name.toLowerCase();
		if (name.endsWith('.csv')) {
			const text = await file.text();
			return parseCSV(text);
		}

		if (name.endsWith('.xlsx') || name.endsWith('.xls')) {
			const data = await file.arrayBuffer();
			const wb = XLSX.read(data, { type: 'array' });
			const first = wb.SheetNames[0];
			if (!first) throw new Error('Workbook contains no sheets');
			const sheet = wb.Sheets[first];
			return XLSX.utils.sheet_to_json(sheet, { header: 1, raw: true, blankrows: false });
		}

		throw new Error('Unsupported file type. Use .csv, .xlsx, or .xls');
	}

	async function handleDatasetFile(e) {
		const file = e.target.files?.[0];
		if (!file) return;
		error = null;
		success = null;
		try {
			const rows = await parseTabularFile(file);
			const { examples, error: parseErr } = rowsToExamples(rows);
			if (parseErr) {
				error = parseErr;
				return;
			}
			const name = newName.trim() || file.name.replace(/\.[^.]+$/, '');
			await api.createDataset(name, newDesc || `Imported from ${file.name}`, examples);
			await loadList();
			await selectDataset(name);
			newName = '';
			newDesc = '';
			importOpen = false;
			e.target.value = '';
		} catch (err) {
			error = err.message;
		}
	}

	function downloadCsv() {
		if (!selected) return;
		const csv = examplesToCSV(currentExamples);
		const blob = new Blob([csv], { type: 'text/csv' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = `${selected}.csv`;
		a.click();
		URL.revokeObjectURL(url);
	}

	onMount(loadList);
</script>

<h1>Datasets</h1>
<p class="muted">
	Create, import (CSV / Excel-exported CSV), edit, and export datasets. Rows are
	<span class="mono">[input_0 … input_n, output]</span> tuples.
</p>

{#if error}<div class="error">{error}</div>{/if}
{#if success}<div class="success">{success}</div>{/if}

<div class="cols">
	<div class="left card">
		<div class="row">
			<h2 class="grow-title">Saved</h2>
			<button onclick={loadList}>Refresh</button>
		</div>

		{#if Object.keys(datasets).length === 0}
			<p class="muted small">None yet.</p>
		{:else}
			<ul class="ds-list">
				{#each Object.entries(datasets) as [name, meta]}
					<li>
						<button
							class="ds-item"
							class:selected={selected === name}
							onclick={() => selectDataset(name)}
						>
							<div class="ds-name mono">{name}</div>
							<div class="muted small">{meta.num_examples} rows</div>
							{#if meta.description}<div class="small">{meta.description}</div>{/if}
						</button>
					</li>
				{/each}
			</ul>
		{/if}

		<div class="stack top-gap">
			<button onclick={() => (createOpen = !createOpen)}>
				{createOpen ? '▼' : '▶'} Create empty
			</button>
			{#if createOpen}
				<label>
					Name <input type="text" bind:value={newName} placeholder="my_dataset" />
				</label>
				<label>
					Description <input type="text" bind:value={newDesc} placeholder="optional" />
				</label>
				<button class="primary" onclick={createNew}>Create</button>
			{/if}

			<button onclick={() => (importOpen = !importOpen)}>
				{importOpen ? '▼' : '▶'} Import CSV
			</button>
			{#if importOpen}
				<p class="muted small">
					Expected columns: <span class="mono">input_0, input_1, …, output</span>. Excel users:
					upload .xlsx/.xls directly, or Save As CSV.
				</p>
				<label>
					Override name <input type="text" bind:value={newName} placeholder="(uses filename)" />
				</label>
				<input
					type="file"
					accept=".csv,.xlsx,.xls,text/csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.ms-excel"
					onchange={handleDatasetFile}
				/>
			{/if}
		</div>
	</div>

	<div class="right">
		{#if selected}
			<div class="card">
				<div class="row">
					<h2 class="grow-title">
						<span class="mono">{selected}</span>
					</h2>
					<button onclick={downloadCsv}>Export CSV</button>
					<a
						class="button-like"
						href={api.exportCsvUrl(selected)}
						target="_blank"
						rel="noreferrer"
					>
						Server CSV
					</a>
					<button class="danger" onclick={deleteSelected}>Delete</button>
				</div>
				<label>
					Description
					<input type="text" bind:value={currentDescription} />
				</label>

				<DatasetEditor bind:examples={currentExamples} bind:numInputs />

				<div class="row top-gap">
					<button class="primary" onclick={saveCurrent} disabled={loading}>Save changes</button>
					<span class="muted small">
						Saving overwrites "{selected}" with current rows.
					</span>
				</div>
			</div>
		{:else}
			<div class="card">
				<p class="muted">Select a dataset on the left — or create/import one.</p>
			</div>
		{/if}
	</div>
</div>

<style>
	.cols {
		display: grid;
		grid-template-columns: 280px 1fr;
		gap: 1rem;
		align-items: start;
	}

	@media (max-width: 800px) {
		.cols {
			grid-template-columns: 1fr;
		}
	}

	.ds-list {
		list-style: none;
		padding: 0;
		margin: 0;
		display: flex;
		flex-direction: column;
		gap: 0.35rem;
	}

	.ds-item {
		width: 100%;
		text-align: left;
		display: block;
		padding: 0.5rem 0.75rem;
	}

	.ds-item.selected {
		border-color: var(--accent);
		background: var(--accent-dim);
	}

	.ds-name {
		font-weight: 600;
		margin-bottom: 0.15rem;
	}

	.grow-title {
		flex: 1;
		margin: 0;
	}

	.top-gap {
		margin-top: 1rem;
	}

	label {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
		font-size: 0.9rem;
		color: var(--text-dim);
	}

	a.button-like {
		display: inline-block;
		padding: 0.45rem 0.9rem;
		background: var(--bg-elev);
		color: var(--text);
		border: 1px solid var(--border);
		border-radius: 6px;
		text-decoration: none;
	}

	a.button-like:hover {
		background: var(--bg-hover);
	}
</style>
