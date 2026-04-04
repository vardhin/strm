<script>
    import { onMount } from 'svelte';
    import { api, activeModel, models, refreshModels, notify } from '$lib/store.js';
    import Spinner from '$lib/components/Spinner.svelte';

    let loading = false;
    let selected = null;
    let detail = null;
    let detailLoading = false;
    let saveLoading = false;
    let deleteLoading = false;

    async function load() {
        loading = true;
        try { await refreshModels(); }
        catch (e) { notify(e.message, 'error'); }
        finally { loading = false; }
    }

    onMount(load);

    async function selectModel(name) {
        selected = name;
        detail = null;
        detailLoading = true;
        try {
            detail = await api(`/models/${name}`);
        } catch (e) {
            notify(e.message, 'error');
        } finally {
            detailLoading = false;
        }
    }

    async function saveModel(name) {
        saveLoading = true;
        try {
            await api(`/models/${name}/save`, { method: 'POST' });
            notify(`Saved '${name}'`, 'success');
        } catch (e) { notify(e.message, 'error'); }
        finally { saveLoading = false; }
    }

    async function deleteModel(name) {
        if (!confirm(`Unload model '${name}'? (Checkpoint files are kept)`)) return;
        deleteLoading = true;
        try {
            await api(`/models/${name}`, { method: 'DELETE' });
            if (selected === name) { selected = null; detail = null; }
            if ($activeModel === name) activeModel.set('default');
            notify(`Unloaded '${name}'`, 'success');
            await load();
        } catch (e) { notify(e.message, 'error'); }
        finally { deleteLoading = false; }
    }

    let modelList = $derived(Object.entries($models));

    function fmtNum(n) { return n.toLocaleString(); }
</script>

<div class="page-header">
    <h1>Models</h1>
    <button class="btn-ghost" onclick={load} disabled={loading}>
        {#if loading}<Spinner />{:else}Refresh{/if}
    </button>
</div>

<div class="layout">
    <div class="list-panel panel">
        <h2>Loaded Models ({modelList.length})</h2>
        {#if modelList.length === 0}
            <p class="empty">No models</p>
        {:else}
            {#each modelList as [name, info]}
                <div class="model-row" class:active={selected === name}>
                    <button class="model-btn" onclick={() => selectModel(name)}>
                        <strong>{name}</strong>
                        <span class="meta">vocab: {info.vocab_size} · {info.num_functions} funcs · {info.train_history_count} runs</span>
                    </button>
                    <div class="actions">
                        <button class="btn-ghost" style="padding:0.2rem 0.5rem;font-size:0.75rem;"
                            onclick={() => saveModel(name)} disabled={saveLoading}>Save</button>
                        {#if name !== 'default'}
                            <button class="btn-danger" style="padding:0.2rem 0.5rem;font-size:0.75rem;"
                                onclick={() => deleteModel(name)} disabled={deleteLoading}>Unload</button>
                        {/if}
                    </div>
                </div>
            {/each}
        {/if}
    </div>

    <div class="detail">
        {#if selected && detailLoading}
            <div class="panel"><Spinner /></div>
        {:else if detail}
            <div class="panel">
                <div class="detail-header">
                    <h2>{detail.model_name}</h2>
                    <span class="tag tag-blue">{fmtNum(detail.total_params)} params</span>
                </div>

                <div class="stats-row">
                    <div class="stat"><div class="sv">{detail.vocab_size}</div><div class="sl">Vocab</div></div>
                    <div class="stat"><div class="sv">{detail.d_model}</div><div class="sl">d_model</div></div>
                    <div class="stat"><div class="sv">{detail.n_layers}</div><div class="sl">Layers</div></div>
                    <div class="stat"><div class="sv">{detail.n_recursions}</div><div class="sl">Recursions</div></div>
                    <div class="stat"><div class="sv">{detail.T}</div><div class="sl">T</div></div>
                </div>

                <h2 style="margin-top:0.5rem">Functions</h2>
                <table>
                    <thead><tr><th>ID</th><th>Name</th><th>Arity</th><th>Layer</th><th>Composition</th></tr></thead>
                    <tbody>
                        {#each detail.functions as f}
                            <tr>
                                <td><code>{f.id}</code></td>
                                <td>{f.name}</td>
                                <td>{f.arity}</td>
                                <td>{f.layer}</td>
                                <td style="font-size:0.75rem;color:#475569">
                                    {#if f.composition}
                                        {f.composition.map(c => `${c.func_name}(${c.args})`).join(' → ')}
                                    {:else}—{/if}
                                </td>
                            </tr>
                        {/each}
                    </tbody>
                </table>

                {#if detail.train_history.length > 0}
                    <h2 style="margin-top:0.5rem">Training History</h2>
                    <table>
                        <thead><tr><th>Target</th><th>Dataset</th><th>Result</th><th>Time (s)</th></tr></thead>
                        <tbody>
                            {#each detail.train_history as h}
                                <tr>
                                    <td>{h.target ?? '—'}</td>
                                    <td>{h.dataset ?? '—'}</td>
                                    <td>
                                        {#if h.success === true}
                                            <span class="tag tag-green">ok</span>
                                        {:else if h.success === false}
                                            <span class="tag tag-red">fail</span>
                                        {:else}
                                            <span class="tag tag-blue">{h.status ?? '?'}</span>
                                        {/if}
                                    </td>
                                    <td>{h.elapsed_s ?? '—'}</td>
                                </tr>
                            {/each}
                        </tbody>
                    </table>
                {/if}
            </div>
        {:else}
            <div class="panel empty">Select a model to view details</div>
        {/if}
    </div>
</div>

<style>
    .layout {
        display: grid;
        grid-template-columns: 280px 1fr;
        gap: 1rem;
        align-items: start;
    }
    .panel {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.25rem;
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
    }
    .model-row {
        display: flex; align-items: center; justify-content: space-between;
        padding: 0.4rem 0.5rem; border-radius: 5px;
        border: 1px solid transparent;
    }
    .model-row.active { background: #0f172a; border-color: #334155; }
    .model-row:hover  { background: #131f34; }
    .model-btn {
        background: none; border: none; color: inherit;
        text-align: left; cursor: pointer; flex: 1;
        display: flex; flex-direction: column; gap: 0.1rem;
    }
    .model-btn strong { color: #e2e8f0; }
    .meta { font-size: 0.72rem; color: #475569; }
    .actions { display: flex; gap: 0.3rem; }

    .detail-header { display: flex; align-items: center; gap: 0.75rem; }
    .stats-row { display: flex; gap: 1rem; flex-wrap: wrap; }
    .stat { text-align: center; min-width: 50px; }
    .sv { font-size: 1.2rem; font-weight: 700; color: #60a5fa; }
    .sl { font-size: 0.7rem; color: #475569; text-transform: uppercase; }
</style>
