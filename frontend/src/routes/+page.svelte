<script>
    import { onMount } from 'svelte';
    import { api, activeModel, models, datasets, experiments,
             refreshModels, refreshDatasets, refreshExperiments, notify } from '$lib/store.js';
    import Spinner from '$lib/components/Spinner.svelte';

    let health = null;
    let loading = true;

    onMount(async () => {
        try {
            health = await api('/health');
            await Promise.all([refreshModels(), refreshDatasets(), refreshExperiments()]);
        } catch (e) {
            notify('Backend unreachable — is the server running?', 'error');
        } finally {
            loading = false;
        }
    });

    let modelList = $derived(Object.entries($models));
    let datasetList = $derived(Object.entries($datasets));
    let recentExps = $derived([...$experiments].reverse().slice(0, 5));
</script>

<div class="page-header">
    <h1>Dashboard</h1>
    {#if loading}<Spinner />{/if}
</div>

{#if health}
    <div class="status-bar">
        <span class="dot green"></span> Backend online
        &nbsp;·&nbsp; {health.loaded_models.length} model{health.loaded_models.length !== 1 ? 's' : ''} loaded
        &nbsp;·&nbsp; {health.datasets.length} dataset{health.datasets.length !== 1 ? 's' : ''}
        &nbsp;·&nbsp; {health.experiments_run} experiment{health.experiments_run !== 1 ? 's' : ''} run
    </div>
{:else if !loading}
    <div class="status-bar offline">
        <span class="dot red"></span> Backend offline
    </div>
{/if}

<div class="grid-3" style="margin-top:1.5rem">
    <a class="stat-card" href="/models">
        <div class="stat-num">{modelList.length}</div>
        <div class="stat-label">Models</div>
    </a>
    <a class="stat-card" href="/datasets">
        <div class="stat-num">{datasetList.length}</div>
        <div class="stat-label">Datasets</div>
    </a>
    <a class="stat-card" href="/experiments">
        <div class="stat-num">{$experiments.length}</div>
        <div class="stat-label">Experiments</div>
    </a>
</div>

<div class="grid-2" style="margin-top:1.5rem; align-items:start">
    <div class="panel">
        <h2>Loaded Models</h2>
        {#if modelList.length === 0}
            <p class="empty">No models loaded</p>
        {:else}
            <table>
                <thead><tr>
                    <th>Name</th><th>Vocab</th><th>Functions</th><th>History</th>
                </tr></thead>
                <tbody>
                    {#each modelList as [name, info]}
                        <tr>
                            <td><a href="/models/{name}">{name}</a></td>
                            <td>{info.vocab_size}</td>
                            <td>{info.num_functions}</td>
                            <td>{info.train_history_count}</td>
                        </tr>
                    {/each}
                </tbody>
            </table>
        {/if}
    </div>

    <div class="panel">
        <h2>Recent Experiments</h2>
        {#if recentExps.length === 0}
            <p class="empty">No experiments yet</p>
        {:else}
            <table>
                <thead><tr>
                    <th>Target</th><th>Model</th><th>Result</th>
                </tr></thead>
                <tbody>
                    {#each recentExps as exp}
                        <tr>
                            <td>{exp.target}</td>
                            <td>{exp.model_name}</td>
                            <td>
                                {#if exp.success}
                                    <span class="tag tag-green">ok</span>
                                {:else}
                                    <span class="tag tag-red">fail</span>
                                {/if}
                            </td>
                        </tr>
                    {/each}
                </tbody>
            </table>
        {/if}
    </div>
</div>

<style>
    .status-bar {
        background: #0f2d1a;
        border: 1px solid #166534;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-size: 0.82rem;
        color: #86efac;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .status-bar.offline { background: #2d0f0f; border-color: #7f1d1d; color: #fca5a5; }
    .dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
    .dot.green { background: #22c55e; }
    .dot.red   { background: #ef4444; }

    .stat-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.25rem;
        text-decoration: none;
        text-align: center;
        transition: background 0.15s;
    }
    .stat-card:hover { background: #273548; }
    .stat-num   { font-size: 2rem; font-weight: 700; color: #60a5fa; }
    .stat-label { font-size: 0.8rem; color: #64748b; margin-top: 0.2rem; text-transform: uppercase; letter-spacing: 0.05em; }

    .panel {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.25rem;
    }
    .panel h2 { margin-bottom: 0.75rem; }
</style>
