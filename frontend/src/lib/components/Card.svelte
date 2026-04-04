<script>
    let { title = '', class: cls = '' } = $props();
</script>

<div class="card {cls}">
    {#if title}
        <h3 class="card-title">{title}</h3>
    {/if}
    <div class="card-body">
        {@render body?.()}
    </div>
</div>

{#snippet body()}
    <slot />
{/snippet}

<style>
    .card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.25rem;
    }
    .card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0 0 0.85rem 0;
    }
    .card-body {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
    }
</style>
