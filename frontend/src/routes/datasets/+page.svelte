<script>
    import { onMount } from 'svelte';
    import { api, datasets, refreshDatasets, notify } from '$lib/store.js';
    import Spinner from '$lib/components/Spinner.svelte';

    let loading = false;
    let selected = null;
    let selectedData = null;
    let detailLoading = false;

    // Create form
    let createName = '';
    let createDesc = '';
    let createExamplesRaw = '';
    let createError = '';
    let createLoading = false;

    // From-function form
    let ffName = '';
    let ffDesc = '';
    let ffFuncId = '';
    let ffInputsRaw = '';
    let ffError = '';
    let ffLoading = false;

    async function load() {
        loading = true;
        try { await refreshDatasets(); }
        catch (e) { notify(e.message, 'error'); }
        finally { loading = false; }
    }

    onMount(load);

    async function selectDataset(name) {
        selected = name;
        selectedData = null;
        detailLoading = true;
        try {
            selectedData = await api(`/datasets/${name}`);
        } catch (e) {
            notify(e.message, 'error');
        } finally {
            detailLoading = false;
        }
    }

    async function deleteDataset(name) {
        if (!confirm(`Delete dataset '${name}'?`)) return;
        try {
            await api(`/datasets/${name}`, { method: 'DELETE' });
            if (selected === name) { selected = null; selectedData = null; }
            notify(`Deleted '${name}'`, 'success');
            await load();
        } catch (e) { notify(e.message, 'error'); }
    }

    async function doCreate() {
        createError = ''; createLoading = true;
        try {
            const examples = JSON.parse(createExamplesRaw);
            await api('/datasets', {
                method: 'POST',
                body: JSON.stringify({ name: createName, description: createDesc, examples }),
            });
            notify(`Dataset '${createName}' created`, 'success');
            createName = ''; createDesc = ''; createExamplesRaw = '';
            await load();
        } catch (e) {
            createError = e.message;
        } finally {
            createLoading = false;
        }
    }

    async function doFromFunction() {
        ffError = ''; ffLoading = true;
        try {
            const input_sets = JSON.parse(ffInputsRaw);
            await api('/datasets/from_function', {
                method: 'POST',
                body: JSON.stringify({
                    name: ffName, description: ffDesc,
                    func_id: parseInt(ffFuncId), input_sets,
                }),
            });
            notify(`Dataset '${ffName}' created from function`, 'success');
            ffName = ''; ffDesc = ''; ffFuncId = ''; ffInputsRaw = '';
            await load();
        } catch (e) {
            ffError = e.message;
        } finally {
            ffLoading = false;
        }
    }

    let datasetList = $derived(Object.entries($datasets));
</script>

<div class="page-header">
    <h1>Datasets</h1>
    <button class="btn-ghost" onclick={load} disabled={loading}>
        {#if loading}<Spinner />{:else}Refresh{/if}
    </button>
</div>

<div class="layout">
    <div class="left">
        <div class="panel">
            <h2>All Datasets ({datasetList.length})</h2>
            {#if datasetList.length === 0}
                <p class="empty">No datasets yet</p>
            {:else}
                {#each datasetList as [name, info]}
                    <div class="ds-row" class:active={selected === name}>
                        <button class="ds-name" onclick={() => selectDataset(name)}>
                            <strong>{name}</strong>
                            <span class="count">{info.num_examples} examples</span>
                        </button>
                        <div class="ds-actions">
                            <a class="btn-ghost" style="padding:0.2rem 0.5rem; font-size:0.75rem; text-decoration:none; border:1px solid #334155; border-radius:4px;"
                               href="http://localhost:8000/datasets/{name}/csv" target="_blank">CSV</a>
                            <button class="btn-danger" style="padding:0.2rem 0.5rem; font-size:0.75rem;"
                                onclick={() => deleteDataset(name)}>Del</button>
                        </div>
                    </div>
                {/each}
            {/if}
        </div>

        {#if selected}
            <div class="panel">
                <h2>{selected}</h2>
                {#if detailLoading}
                    <Spinner />
                {:else if selectedData}
                    {#if selectedData.description}
                        <p style="color:#64748b;font-size:0.8rem;">{selectedData.description}</p>
                    {/if}
                    <table>
                        <thead><tr><th>Inputs</th><th>Output</th></tr></thead>
                        <tbody>
                            {#each selectedData.examples.slice(0, 50) as [inputs, output]}
                                <tr>
                                    <td><code>{JSON.stringify(inputs)}</code></td>
                                    <td><code>{output}</code></td>
                                </tr>
                            {/each}
                        </tbody>
                    </table>
                    {#if selectedData.examples.length > 50}
                        <p style="color:#64748b;font-size:0.75rem;">… {selectedData.examples.length - 50} more</p>
                    {/if}
                {/if}
            </div>
        {/if}
    </div>

    <div class="side">
        <div class="panel">
            <h2>Create from Examples</h2>
            <div class="field">
                <label>Name</label>
                <input bind:value={createName} placeholder="e.g. add_pairs" />
            </div>
            <div class="field">
                <label>Description (optional)</label>
                <input bind:value={createDesc} placeholder="What does this dataset test?" />
            </div>
            <div class="field">
                <label>Examples JSON <small>([[inputs], output] pairs)</small></label>
                <textarea rows="6" bind:value={createExamplesRaw}
                    placeholder={`[[[1,2],3],\n [[3,4],7],\n [[0,5],5]]`}></textarea>
            </div>
            {#if createError}<p class="error-msg">{createError}</p>{/if}
            <button class="btn-primary" onclick={doCreate} disabled={createLoading || !createName}>
                {#if createLoading}<Spinner />{:else}Create Dataset{/if}
            </button>
        </div>

        <div class="panel">
            <h2>Generate from Function</h2>
            <div class="field">
                <label>Name</label>
                <input bind:value={ffName} placeholder="e.g. add_generated" />
            </div>
            <div class="field">
                <label>Function ID</label>
                <input type="number" bind:value={ffFuncId} placeholder="e.g. 2" />
            </div>
            <div class="field">
                <label>Input sets JSON</label>
                <textarea rows="5" bind:value={ffInputsRaw}
                    placeholder={`[[1,2],[3,4],[5,6]]`}></textarea>
            </div>
            {#if ffError}<p class="error-msg">{ffError}</p>{/if}
            <button class="btn-primary" onclick={doFromFunction} disabled={ffLoading || !ffName || !ffFuncId}>
                {#if ffLoading}<Spinner />{:else}Generate{/if}
            </button>
        </div>
    </div>
</div>

<style>
    .layout {
        display: grid;
        grid-template-columns: 1fr 320px;
        gap: 1rem;
        align-items: start;
    }
    .left, .side { display: flex; flex-direction: column; gap: 1rem; }
    .panel {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.25rem;
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
    }
    .ds-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.4rem 0.5rem;
        border-radius: 5px;
        border: 1px solid transparent;
    }
    .ds-row.active { background: #0f172a; border-color: #334155; }
    .ds-row:hover  { background: #131f34; }
    .ds-name {
        background: none; border: none; color: inherit;
        text-align: left; cursor: pointer; flex: 1;
        display: flex; flex-direction: column; gap: 0.1rem;
    }
    .ds-name strong { color: #e2e8f0; }
    .count { font-size: 0.75rem; color: #475569; }
    .ds-actions { display: flex; gap: 0.3rem; align-items: center; }
    textarea { resize: vertical; font-family: monospace; }
</style>
