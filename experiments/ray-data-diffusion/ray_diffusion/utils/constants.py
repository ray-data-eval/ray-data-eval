"""Define constants here."""

# Keys in Ray Dataset
IMAGE_KEY = "image"

CAPTION_KEY = "caption"

IMAGE_LATENTS_256_KEY = "latents_256_bytes"

IMAGE_LATENTS_512_KEY = "latents_512_bytes"

CAPTION_LATENTS_KEY = "caption_latents"

SDXL_IMAGE_LATENTS_256_KEY = "latents_256_sdxl"

SDXL_IMAGE_LATENTS_512_KEY = "latents_512_sdxl"

SDXL_CAPTION_LATENTS_1_KEY = "caption_latents_1_sdxl"

SDXL_CAPTION_LATENTS_2_KEY = "caption_latents_2_sdxl"

SDXL_ADD_TEXT_EMBEDS_KEY = "add_text_embeds_sdxl"

CROP_AND_SIZE_CONDITIONING_KEY = "crop_and_size_conditioning"

TENSOR_KEY_LIST = [
        IMAGE_LATENTS_256_KEY,
        IMAGE_LATENTS_512_KEY,
        CAPTION_LATENTS_KEY,
        SDXL_IMAGE_LATENTS_256_KEY,
        SDXL_IMAGE_LATENTS_512_KEY,
        SDXL_CAPTION_LATENTS_1_KEY,
        SDXL_CAPTION_LATENTS_2_KEY,
        SDXL_ADD_TEXT_EMBEDS_KEY,
        CROP_AND_SIZE_CONDITIONING_KEY
]

PROMPT_SAMPLES = [
    "Christmas Shopping Copy",
    "Virgina Grist Mill in Autumn",
    "Lavender Tea Service",
    "1946 Ford Convertible Street Rod - 3 - Print Image",
    "christmas, winter, and marshmallow image",
    "Twilight city skyline with glowing skyscrapers.",
    "Snowy village with festive lights and a Christmas tree.",
    "Dusty sunbeams in an ancient, book-filled library.",
    "Cherry blossoms over a tranquil Japanese koi pond.",
    "Elephants at a watering hole on a sunset savannah.",
    "Sunny farmer's market with fresh produce and bustling shoppers.",
    "Dawn at a misty mountain lake with pine trees.",
    "Rain-soaked cobblestone street lit by old lamps.",
    "Colorful coral reef with fish under sunlit water.",
    "Jazz band playing in a lively 1920s speakeasy.",
]
