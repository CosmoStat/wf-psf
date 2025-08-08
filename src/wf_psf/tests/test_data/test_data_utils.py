
class MockDataset:
    def __init__(self, positions, zernike_priors, star_type, stars, masks):
        self.dataset = {"positions": positions, "zernike_prior": zernike_priors, star_type: stars, "masks": masks}
        
class MockData:
    def __init__(
        self,
        training_positions,
        test_positions,
        training_zernike_priors=None,
        test_zernike_priors=None,
        noisy_stars=None,
        noisy_masks=None,
        stars=None,
        masks=None,
    ):
        self.training_data = MockDataset(
            positions=training_positions, 
            zernike_priors=training_zernike_priors,
            star_type="noisy_stars",
            stars=noisy_stars,
            masks=noisy_masks)
        self.test_data = MockDataset(
            positions=test_positions, 
            zernike_priors=test_zernike_priors,
            star_type="stars",
            stars=stars,
            masks=masks)

