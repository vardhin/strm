<script>
    import { page } from '$app/stores';
    import { activeModel, models, refreshModels, notify } from '$lib/store.js';
    import { onMount } from 'svelte';

    const links = [
        { href: '/',            label: 'Dashboard' },
        { href: '/registry',    label: 'Registry'  },
        { href: '/datasets',    label: 'Datasets'  },
        { href: '/training',    label: 'Training'  },
        { href: '/models',      label: 'Models'    },
        { href: '/testing',     label: 'Testing'   },
        { href: '/experiments', label: 'Experiments' },
    ];

    onMount(async () => {
        try {
            await refreshModels();
        } catch (e) {
            notify('Could not reach backend', 'error');
        }
    });

    let modelNames = $derived(Object.keys($models));
</script>

<nav>
    <a class="brand" href="/">NSSR</a>

    <ul class="links">
        {#each links as { href, label }}
            <li>
                <a
                    {href}
                    class:active={$page.url.pathname === href}
                >{label}</a>
            </li>
        {/each}
    </ul>

    <div class="model-select">
        <label for="model-pick">Model:</label>
        <select id="model-pick" bind:value={$activeModel}>
            {#each modelNames as m}
                <option value={m}>{m}</option>
            {/each}
            {#if !modelNames.includes($activeModel)}
                <option value={$activeModel}>{$activeModel}</option>
            {/if}
        </select>
    </div>
</nav>

<style>
    nav {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0 1.5rem;
        height: 52px;
        background: #0f172a;
        border-bottom: 1px solid #1e293b;
        position: sticky;
        top: 0;
        z-index: 100;
    }
    .brand {
        font-weight: 700;
        font-size: 1.1rem;
        color: #60a5fa;
        text-decoration: none;
        letter-spacing: 0.05em;
        margin-right: 0.5rem;
    }
    ul.links {
        display: flex;
        gap: 0.25rem;
        list-style: none;
        margin: 0;
        padding: 0;
        flex: 1;
    }
    ul.links a {
        display: block;
        padding: 0.3rem 0.7rem;
        border-radius: 5px;
        color: #94a3b8;
        text-decoration: none;
        font-size: 0.875rem;
        transition: background 0.15s, color 0.15s;
    }
    ul.links a:hover  { background: #1e293b; color: #e2e8f0; }
    ul.links a.active { background: #1d4ed8; color: #fff; }

    .model-select {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.8rem;
        color: #64748b;
    }
    select {
        background: #1e293b;
        color: #e2e8f0;
        border: 1px solid #334155;
        border-radius: 4px;
        padding: 0.25rem 0.5rem;
        font-size: 0.8rem;
        cursor: pointer;
    }
</style>
