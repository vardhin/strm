<script>
    import { toast } from '$lib/store.js';
</script>

{#if $toast}
    <div class="toast toast-{$toast.type}">
        {$toast.message}
    </div>
{/if}

<style>
    .toast {
        position: fixed;
        bottom: 1.5rem;
        right: 1.5rem;
        padding: 0.75rem 1.25rem;
        border-radius: 6px;
        font-size: 0.9rem;
        font-weight: 500;
        z-index: 9999;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        animation: slide-in 0.2s ease;
    }
    .toast-info    { background: #3b82f6; color: #fff; }
    .toast-success { background: #22c55e; color: #fff; }
    .toast-error   { background: #ef4444; color: #fff; }
    @keyframes slide-in {
        from { transform: translateY(1rem); opacity: 0; }
        to   { transform: translateY(0);    opacity: 1; }
    }
</style>
