import torch
import torch.nn.functional as F
from typing import List, Tuple, Any, Optional, Dict
from symbolic import SymbolicRegistry, ProgramExecutor

class ProgramSearcher:
    """Handles program search using TRM and exhaustive methods."""
    
    def __init__(self, registry: SymbolicRegistry, executor: ProgramExecutor, 
                 model, config: Dict, training_history: List):
        self.registry = registry
        self.executor = executor
        self.model = model
        self.config = config
        self.training_history = training_history
    
    def exhaustive_search(self, examples: List[Tuple[List[int], Any]]) -> Optional[Dict]:
        """Exhaustive search with 3-function compositions."""
        vocab_size = self.registry.get_vocab_size()
        
        # Try single operations
        for primary_id in range(vocab_size):
            if self.executor.validate_program(primary_id, 0, 'none', examples):
                return {
                    'primary_id': primary_id,
                    'secondary_id': 0,
                    'tertiary_id': None,
                    'comp_type': 'none',
                    'score': 1.0,
                    'step': 0
                }
        
        # Try 2-function compositions
        for primary_id in range(vocab_size):
            for secondary_id in range(vocab_size):
                for comp_type in ['sequential', 'nested']:
                    if self.executor.validate_program(primary_id, secondary_id, comp_type, examples):
                        return {
                            'primary_id': primary_id,
                            'secondary_id': secondary_id,
                            'tertiary_id': None,
                            'comp_type': comp_type,
                            'score': 1.0,
                            'step': 0
                        }
        
        # Try 3-function parallel composition
        for primary_id in range(vocab_size):
            for secondary_id in range(vocab_size):
                for tertiary_id in range(vocab_size):
                    if self.executor.validate_program(primary_id, secondary_id, 'parallel', examples, tertiary_id):
                        return {
                            'primary_id': primary_id,
                            'secondary_id': secondary_id,
                            'tertiary_id': tertiary_id,
                            'comp_type': 'parallel',
                            'score': 1.0,
                            'step': 0
                        }
        
        return None
    
    def trm_search(self, examples: List[Tuple[List[int], Any]], 
                   task_name: str, x_input: torch.Tensor, carry,
                   exploration_bonus: float = 0.5) -> Optional[Dict]:
        """TRM-based learned search with exploration."""
        print(f"  [TRM Search] Using learned heuristics (vocab size = {self.registry.get_vocab_size()})...")
        
        self.model.eval()
        candidates = []
        
        # Build layer-based priors
        layer_functions = {}
        for fid, meta in self.registry.metadata.items():
            layer = meta['layer']
            if layer not in layer_functions:
                layer_functions[layer] = []
            layer_functions[layer].append(fid)
        
        # Compute recent usage penalties
        recent_usage = {}
        for i, history_item in enumerate(self.training_history[-5:]):
            prog = history_item['program']
            decay = (len(self.training_history[-5:]) - i) / len(self.training_history[-5:])
            recent_usage[prog['primary_id']] = recent_usage.get(prog['primary_id'], 0) + decay
            recent_usage[prog['secondary_id']] = recent_usage.get(prog['secondary_id'], 0) + decay
            if prog.get('tertiary_id') is not None:
                recent_usage[prog['tertiary_id']] = recent_usage.get(prog['tertiary_id'], 0) + decay
        
        self._print_search_context(layer_functions, recent_usage, exploration_bonus)
        
        with torch.no_grad():
            for step in range(self.config['halt_max_steps']):
                carry, outputs = self.model(carry, x_input)
                
                primary_probs = F.softmax(outputs['primary_logits'], dim=-1)
                secondary_probs = F.softmax(outputs['secondary_logits'], dim=-1)
                comp_probs = F.softmax(outputs['composition_logits'], dim=-1)
                
                # Apply exploration bonus
                primary_probs_adjusted, secondary_probs_adjusted = self._apply_exploration_bonus(
                    primary_probs, secondary_probs, recent_usage, exploration_bonus
                )
                
                self._print_step_predictions(step, outputs, primary_probs_adjusted, 
                                            secondary_probs_adjusted, comp_probs, 
                                            primary_probs, secondary_probs, recent_usage)
                
                # Generate candidates
                candidates.extend(self._generate_candidates(
                    primary_probs_adjusted, secondary_probs_adjusted, comp_probs, outputs, step
                ))
                
                if outputs['confidence'][0] > 0.8 or step >= 5:
                    print(f"\n    [Halting] Confidence: {outputs['confidence'][0].item():.4f} (threshold: 0.8, step: {step+1}/5)")
                    break
        
        # Validate candidates
        return self._validate_candidates(candidates, examples)
    
    def _apply_exploration_bonus(self, primary_probs, secondary_probs, recent_usage, exploration_bonus):
        """Apply exploration penalty to recent predictions."""
        primary_probs_adjusted = primary_probs.clone()
        secondary_probs_adjusted = secondary_probs.clone()
        
        for fid, count in recent_usage.items():
            penalty = min(exploration_bonus * count, 0.9)
            primary_probs_adjusted[0, fid] *= (1 - penalty)
            secondary_probs_adjusted[0, fid] *= (1 - penalty)
        
        # Renormalize
        primary_probs_adjusted = primary_probs_adjusted / primary_probs_adjusted.sum()
        secondary_probs_adjusted = secondary_probs_adjusted / secondary_probs_adjusted.sum()
        
        return primary_probs_adjusted, secondary_probs_adjusted
    
    def _generate_candidates(self, primary_probs, secondary_probs, comp_probs, outputs, step):
        """Generate candidate programs from predictions."""
        candidates = []
        vocab_size = self.registry.get_vocab_size()
        top_k = min(vocab_size, 5)
        
        primary_top = torch.topk(primary_probs[0], top_k)
        secondary_top = torch.topk(secondary_probs[0], top_k)
        comp_top = torch.topk(comp_probs[0], min(4, comp_probs.shape[-1]))
        
        # 3-function parallel compositions
        for p_idx, p_prob in zip(primary_top.indices, primary_top.values):
            for s_idx, s_prob in zip(secondary_top.indices, secondary_top.values):
                for t_idx in range(vocab_size):
                    t_prob = primary_probs[0, t_idx]
                    
                    p_layer = self.registry.metadata[p_idx.item()]['layer']
                    s_layer = self.registry.metadata[s_idx.item()]['layer']
                    t_layer = self.registry.metadata[t_idx]['layer']
                    
                    diversity_bonus = 1.5 if len(set([p_layer, s_layer, t_layer])) > 1 else 1.0
                    score = (p_prob * s_prob * t_prob * outputs['confidence'][0] * diversity_bonus).item()
                    
                    candidates.append({
                        'primary_id': p_idx.item(),
                        'secondary_id': s_idx.item(),
                        'tertiary_id': t_idx,
                        'comp_type': 'parallel',
                        'score': score,
                        'step': step
                    })
                
                # 2-function compositions
                for c_idx, c_prob in zip(comp_top.indices, comp_top.values):
                    comp_types = ['none', 'sequential', 'nested', 'parallel']
                    comp_type = comp_types[c_idx.item()]
                    
                    if comp_type != 'parallel':
                        p_layer = self.registry.metadata[p_idx.item()]['layer']
                        s_layer = self.registry.metadata[s_idx.item()]['layer']
                        
                        diversity_bonus = 1.5 if (comp_type != 'none' and p_layer != s_layer) else 1.0
                        score = (p_prob * s_prob * c_prob * outputs['confidence'][0] * diversity_bonus).item()
                        
                        candidates.append({
                            'primary_id': p_idx.item(),
                            'secondary_id': s_idx.item(),
                            'tertiary_id': None,
                            'comp_type': comp_type,
                            'score': score,
                            'step': step
                        })
        
        return candidates
    
    def _validate_candidates(self, candidates, examples):
        """Sort and validate candidates."""
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n    [Validation] Testing top {min(len(candidates), 100)} candidates...")
        print(f"      Top-5 candidates by score:")
        for i, candidate in enumerate(candidates[:5]):
            p_name = self.registry.metadata[candidate['primary_id']]['name']
            s_name = self.registry.metadata[candidate['secondary_id']]['name']
            t_name = self.registry.metadata[candidate['tertiary_id']]['name'] if candidate.get('tertiary_id') else 'N/A'
            print(f"        {i+1}. {p_name} + {s_name} + {t_name} ({candidate['comp_type']}): {candidate['score']:.6f}")
        
        for i, candidate in enumerate(candidates[:100]):
            if i < 10:
                if self._validate_and_print(candidate, examples, i):
                    return candidate
            else:
                if self.executor.validate_program(
                    candidate['primary_id'],
                    candidate['secondary_id'],
                    candidate['comp_type'],
                    examples,
                    candidate.get('tertiary_id')
                ):
                    self._print_success(candidate, i)
                    return candidate
        
        print(f"\n  [Failed] No valid program found in top 100 candidates")
        return None
    
    def _validate_and_print(self, candidate, examples, rank):
        """Validate candidate with detailed printing."""
        p_name = self.registry.metadata[candidate['primary_id']]['name']
        s_name = self.registry.metadata[candidate['secondary_id']]['name']
        t_name = self.registry.metadata[candidate['tertiary_id']]['name'] if candidate.get('tertiary_id') else 'N/A'
        
        print(f"\n      Testing Rank {rank+1}: {p_name} + {s_name} + {t_name} ({candidate['comp_type']})")
        print(f"        Executing on examples:")
        
        all_match = True
        for inp, expected in examples:
            try:
                result = self.executor.execute_program(
                    candidate['primary_id'],
                    candidate['secondary_id'],
                    candidate['comp_type'],
                    inp,
                    candidate.get('tertiary_id')
                )
                match = result == expected
                all_match = all_match and match
                status = "✓" if match else "✗"
                print(f"          {status} f{tuple(inp)} = {result} (expected {expected})")
            except Exception as e:
                print(f"          ✗ f{tuple(inp)} = ERROR: {str(e)[:50]}")
                all_match = False
        
        if all_match:
            self._print_success(candidate, rank)
            return True
        return False
    
    def _print_success(self, candidate, rank):
        """Print success message."""
        p_name = self.registry.metadata[candidate['primary_id']]['name']
        s_name = self.registry.metadata[candidate['secondary_id']]['name']
        t_name = self.registry.metadata[candidate['tertiary_id']]['name'] if candidate.get('tertiary_id') else None
        
        print(f"\n  [Success] Found via TRM at step {candidate['step']} (rank {rank+1}/100)")
        print(f"    Primary: {p_name}")
        print(f"    Secondary: {s_name}")
        if t_name:
            print(f"    Tertiary: {t_name}")
        print(f"    Composition: {candidate['comp_type']}")
        print(f"    Score: {candidate['score']:.6f}")
    
    def _print_search_context(self, layer_functions, recent_usage, exploration_bonus):
        """Print search context."""
        print(f"    Available functions by layer:")
        for layer in sorted(layer_functions.keys(), reverse=True):
            func_names = [self.registry.metadata[fid]['name'] for fid in layer_functions[layer]]
            print(f"      Layer {layer}: {func_names}")
        
        if recent_usage:
            print(f"    Recent usage penalties (decay-weighted):")
            for fid, count in sorted(recent_usage.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"      {self.registry.metadata[fid]['name']}: -{exploration_bonus * count:.3f}")
    
    def _print_step_predictions(self, step, outputs, primary_probs_adjusted, 
                               secondary_probs_adjusted, comp_probs, 
                               primary_probs, secondary_probs, recent_usage):
        """Print TRM predictions for current step."""
        print(f"\n    [Step {step}] TRM Predictions:")
        print(f"      Confidence: {outputs['confidence'][0].item():.4f}")
        
        vocab_size = self.registry.get_vocab_size()
        
        print(f"      Top-3 Primary (after exploration bonus):")
        top_primary = torch.topk(primary_probs_adjusted[0], min(3, vocab_size))
        for i in range(len(top_primary.indices)):
            idx = top_primary.indices[i].item()
            prob = top_primary.values[i].item()
            original_prob = primary_probs[0, idx].item()
            penalty_str = f" (was {original_prob:.4f})" if idx in recent_usage else ""
            print(f"        {i+1}. {self.registry.metadata[idx]['name']}: {prob:.4f}{penalty_str}")
        
        print(f"      Top-3 Secondary (after exploration bonus):")
        top_secondary = torch.topk(secondary_probs_adjusted[0], min(3, vocab_size))
        for i in range(len(top_secondary.indices)):
            idx = top_secondary.indices[i].item()
            prob = top_secondary.values[i].item()
            original_prob = secondary_probs[0, idx].item()
            penalty_str = f" (was {original_prob:.4f})" if idx in recent_usage else ""
            print(f"        {i+1}. {self.registry.metadata[idx]['name']}: {prob:.4f}{penalty_str}")
        
        print(f"      Composition probabilities:")
        comp_types = ['none', 'sequential', 'nested', 'parallel']
        for comp_type, prob in zip(comp_types, comp_probs[0]):
            print(f"        {comp_type}: {prob.item():.4f}")