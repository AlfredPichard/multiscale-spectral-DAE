import numpy as np

import src.networks.utils as utils


class EDMSampler:
    def __init__(
            self, 
            num_steps=18, 
            min_noise_level=0.002,
            max_noise_level=80,
            rho=7,
            S_churn=40, 
            S_min=0, 
            S_max=float('inf'),
            S_noise=1.001,
            sigma_data = 0.5,
            cfg_weight=7,
            target_type='audio',
            device='cpu',
        ) -> None:
        assert (target_type in ["audio", "image"]), "target_type must be either audio or image."
        self.num_steps = num_steps
        self.min_noise_level = min_noise_level
        self.max_noise_level = max_noise_level
        self.rho=rho
        self.S_churn=S_churn
        self.S_min=S_min
        self.S_max=S_max
        self.S_noise=S_noise
        self.sigma_data = sigma_data
        self.cfg_weight = cfg_weight
        self.target_type = target_type
        self.device = device

        self.scaler = utils.EDMScaler(self.sigma_data, device=self.device)
    
    def _to_adapted_float_tensor(self, x, size):
        return utils.to_adapted_float_tensor(x, size, self.target_type, self.device)

    @torch.no_grad()
    def sample(
        self, model, x_T, labels=None, z_semantic=None, z_temporal=None, z_hierarchical=None):
        # Define steps
        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=self.device)
        t_steps = (self.max_noise_level ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (self.min_noise_level ** (1 / self.rho) - self.max_noise_level ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
        batch_size = x_T.size(0)

        # Sampling
        x_sequence = []
        x_next = x_T.to(torch.float64) * t_steps[0]
        for i, (t_current, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            # Init
            x_current = x_next
            epsilon_current = torch.randn_like(x_current)*self.S_noise**2

            # Add noise for Stochastic proccess
            gamma_i = min(self.S_churn/self.num_steps, np.sqrt(2)-1) if self.S_min <= t_current.item() <= self.S_max else 0
            t_current_bar = t_current + gamma_i*t_current
            x_current_bar = (x_current + torch.sqrt(t_current_bar**2 - t_current**2)*epsilon_current).float()

            # Conditional model eval
            model_output = self.scaler.eval(model, batch_size, x_current_bar, t_current_bar, labels, z_semantic).float()

            # Classifier-free guidance
            if labels is not None:
                free_model_output = self.scaler.eval(
                    model=model, 
                    batch_size=batch_size, 
                    x=x_current_bar, 
                    sigma=t_current_bar,
                    y=None,
                    z=z_semantic,
                    z_temporal=z_temporal,
                    z_hierarchical=z_hierarchical).float()
                model_output = (1 + self.cfg_weight)*model_output - self.cfg_weight*free_model_output
            
            # Euler step
            t_current_bar = self._to_adapted_float_tensor(t_current_bar, batch_size)
            d = (x_current_bar - model_output)/(t_current_bar)
            x_next = x_current_bar + (t_next - t_current_bar)*d

            # 2nd order correction
            if i < self.num_steps - 1:
                # Conditional model eval
                model_output = self.scaler.eval(model, batch_size, x_next, t_next, labels, z_semantic)
                
                # Classifier-free guidance
                if labels is not None:
                    free_model_output = self.scaler.eval(
                        model=model, 
                        batch_size=batch_size, 
                        x=x_next, 
                        sigma=t_next,
                        y=None,
                        z=z_semantic,
                        z_temporal=z_temporal,
                        z_hierarchical=z_hierarchical).float()                    
                    model_output = (1 + self.cfg_weight)*model_output - self.cfg_weight*free_model_output

                t_next = self._to_adapted_float_tensor(t_next, batch_size)
                d_prime = (x_next - model_output)/(t_next)
                x_next = x_current_bar + (t_next - t_current_bar)*(0.5*d + 0.5*d_prime)

            x_sequence.append(x_next)

        return x_next, x_sequence
    

class AlphaSampler:
    def __init__(
            self, 
            num_steps=18, 
            min_noise_level=0.001,
            max_noise_level=80,
            alpha=1,
            rho=7,
            sigma_data=0.5,
            cfg_weight = 8,
            device='cpu',
        ) -> None:
        self.num_steps = num_steps
        self.min_noise_level = min_noise_level
        self.max_noise_level = max_noise_level
        self.rho=rho
        self.sigma_data =sigma_data
        self.cfg_weight = cfg_weight
        self.alpha = alpha
        self.device = device
        
        self.scaler = utils.EDMScaler(self.sigma_data, device=self.device)

    @torch.no_grad()
    def sample(
        self, model, x_T, labels=None, z_semantic=None, z_temporal=None, z_hierarchical=None):
        # Init
        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=self.device)
        t_steps = (self.max_noise_level ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (self.min_noise_level ** (1 / self.rho) - self.max_noise_level ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
        batch_size = x_T.size(0)

        # Sampling
        x_sequence = []
        x_next = x_T.float() * t_steps[0]
        for i, (t_current, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_current = x_next
            step_length = t_next - t_current

            # Conditional model eval
            model_output = self.scaler.eval(
                model=model, 
                batch_size=batch_size, 
                x=x_current, 
                sigma=t_current,
                y=labels,
                z=z_semantic,
                z_temporal=z_temporal,
                z_hierarchical=z_hierarchical).float()
            
            # Classifier-free guidance
            if labels is not None:
                free_model_output = self.scaler.eval(
                    model=model, 
                    batch_size=batch_size, 
                    x=x_current, 
                    sigma=t_current,
                    y=None,
                    z=z_semantic,
                    z_temporal=z_temporal,
                    z_hierarchical=z_hierarchical).float()

                model_output = (1 + self.cfg_weight)*model_output - self.cfg_weight*free_model_output
                   
            # Euler step
            d_current = (x_current - model_output)/ t_current
            x_current_prime = x_current + self.alpha*step_length*d_current
            t_current_prime = t_current + self.alpha*step_length

            # 2nd order step
            if i < self.num_steps - 1:
                # Conditional model eval
                model_output = self.scaler.eval(
                    model=model, 
                    batch_size=batch_size, 
                    x=x_current_prime, 
                    sigma=t_current_prime, 
                    y=labels, 
                    z=z_semantic,
                    z_temporal=z_temporal,
                    z_hierarchical=z_hierarchical).float()
                
                # Classifier-free guidance
                if labels is not None:
                    free_model_output = self.scaler.eval(
                        model=model, 
                        batch_size=batch_size, 
                        x=x_current_prime, 
                        sigma=t_current_prime,
                        y=None,
                        z=z_semantic,
                        z_temporal=z_temporal,
                        z_hierarchical=z_hierarchical).float()
                    model_output = (1 + self.cfg_weight)*model_output - self.cfg_weight*free_model_output

                d_current_prime = (x_current - model_output)/(t_current_prime)
                x_next = x_current + step_length*((1-1/(2*self.alpha))*d_current + (1/(2*self.alpha))*d_current_prime)

            else:
                x_next = x_current + step_length*d_current

            x_sequence.append(x_next)

        return x_next, x_sequence


### Euler Deterministic order 1 sampler (simplified AlphaSampler)
class EulerSampler:
    def __init__(
            self, 
            num_steps=50, 
            min_noise_level=0.002,
            max_noise_level=80,
            rho=7,
            sigma_data=0.5,
            cfg_weight = 8,
            reverse_num_steps = 1000,
            target_type='audio',
            device='cpu',
        ) -> None:
        assert (target_type in ["audio", "image"]), "target_type must be either audio or image."
        self.num_steps = num_steps
        self.reverse_num_steps = reverse_num_steps
        self.min_noise_level = min_noise_level
        self.max_noise_level = max_noise_level
        self.rho=rho
        self.sigma_data =sigma_data
        self.cfg_weight = cfg_weight
        self.target_type = target_type
        self.device = device

        self.scaler = utils.EDMScaler(self.sigma_data, device=self.device)

    @torch.no_grad()
    def sample(self, model, x_T, labels=None, z_semantic=None, z_temporal=None, z_hierarchical=None):
        # Init
        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=self.device)
        t_steps = (self.max_noise_level ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (self.min_noise_level ** (1 / self.rho) - self.max_noise_level ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
        batch_size = x_T.size(0)

        # Sampling
        x_sequence = []
        x_next = x_T.float() * t_steps[0]
        for _, (t_current, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_current = x_next
            step_length = t_next - t_current

            # Conditional model eval
            model_output = self.scaler.eval(
                model=model, 
                batch_size=batch_size, 
                x=x_current, 
                sigma=t_current,
                y=labels,
                z=z_semantic,
                z_temporal=z_temporal,
                z_hierarchical=z_hierarchical).float()
            
            # Classifier-free guidance
            if labels is not None:
                free_model_output = self.scaler.eval(
                    model=model, 
                    batch_size=batch_size, 
                    x=x_current, 
                    sigma=t_current,
                    y=None,
                    z=z_semantic,
                    z_temporal=z_temporal,
                    z_hierarchical=z_hierarchical).float()
                model_output = (1 + self.cfg_weight)*model_output - self.cfg_weight*free_model_output
            
            d_current = (x_current - model_output)/ t_current
            x_next = x_current + step_length*d_current

            x_sequence.append(x_next)

        return x_next, x_sequence
    
    @torch.no_grad()
    def reverse_sample(self, model, x_0, labels=None, z_semantic=None, z_temporal=None, z_hierarchical=None):
        # Init
        step_indices = torch.arange(self.reverse_num_steps, dtype=torch.float64, device=self.device)
        t_steps = (self.max_noise_level ** (1 / self.rho) + step_indices / (self.reverse_num_steps - 1) * (self.min_noise_level ** (1 / self.rho) - self.max_noise_level ** (1 / self.rho))) ** self.rho
        t_steps = reversed(torch.as_tensor(t_steps))
        batch_size = x_0.size(0)

        # Sampling
        x_sequence = []
        x_next = x_0.float()
        for _, (t_current, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_current = x_next
            step_length = t_next - t_current
            
            # Conditional model eval
            model_output = self.scaler.eval(
                model=model, 
                batch_size=batch_size, 
                x=x_current, 
                sigma=t_next,
                y=labels,
                z=z_semantic,
                z_temporal=z_temporal,
                z_hierarchical=z_hierarchical).float()
            
            # Classifier-free guidance
            if labels is not None:
                free_model_output = self.scaler.eval(
                    model=model, 
                    batch_size=batch_size, 
                    x=x_current, 
                    sigma=t_next,
                    y=None,
                    z=z_semantic,
                    z_temporal=z_temporal,
                    z_hierarchical=z_hierarchical).float()
                model_output = (1 + self.cfg_weight)*model_output - self.cfg_weight*free_model_output
            
            d_current = (x_current - model_output)/ t_current
            x_next = x_current + step_length*d_current

            x_sequence.append(x_next)

        return x_next, x_sequence
