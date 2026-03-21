import os
import logging
from huggingface_hub import snapshot_download
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
import sys

def setup_logging(level=logging.INFO):
    """Setup clean logging format for nano-vllm."""
    formatter = logging.Formatter(
        fmt='%(levelname)s [%(name)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler('nano_vllm.log', mode='w')
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = []  # Clear existing handlers
    root_logger.addHandler(console)
    root_logger.addHandler(file_handler)
    
    # Quiet down some noisy third-party loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# Setup logging
setup_logging(logging.INFO)


def main():
    model_id = "Qwen/Qwen3-0.6B"
    path = "/home/albert/learn/models/Qwen3-0.6B/"
    os.makedirs(path, exist_ok=True)
    if not os.path.isfile(os.path.join(path, "config.json")):
        snapshot_download(
            repo_id=model_id,
            local_dir=path,
            local_dir_use_symlinks=False,
            resume_download=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)  # disable cuda graphs, use eager execution

    sampling_params = SamplingParams(temperature=0.6, max_tokens=1024)

   
    prompts = [
        "In the realm of artificial intelligence and machine learning, neural networks have emerged as a powerful paradigm for solving complex problems. These computational models, inspired by the biological neural networks in animal brains, consist of interconnected layers of artificial neurons that process and transform information through weighted connections. The fundamental building blocks of neural networks include input layers that receive raw data, hidden layers that perform intermediate computations, and output layers that produce final predictions or classifications. Deep learning architectures, which contain multiple hidden layers, have revolutionized fields such as computer vision, natural language processing, and speech recognition. Training these networks typically involves optimization algorithms like gradient descent and backpropagation, which adjust the network's parameters to minimize a loss function that measures the difference between predicted and actual outputs. Regularization techniques such as dropout, batch normalization, and weight decay help prevent overfitting and improve generalization performance. Modern neural network architectures like Transformers have further advanced the field by employing self-attention mechanisms that capture long-range dependencies in sequential data, enabling breakthroughs in language understanding and generation tasks. The scalability of these models has led to the development of large language models with billions of parameters, capable of performing a wide range of natural language tasks with remarkable accuracy and coherence.",
        
        "The history of computing spans several decades of rapid technological advancement, beginning with early mechanical calculators and evolving into today's sophisticated digital systems. The invention of the transistor in the mid-20th century marked a pivotal moment, enabling the development of smaller, faster, and more reliable electronic computers. The subsequent emergence of integrated circuits and microprocessors revolutionized the industry, making personal computers accessible to consumers and businesses alike. The internet and World Wide Web transformed how people communicate and access information, creating new opportunities for collaboration and knowledge sharing. Mobile computing devices and cloud computing have further changed the landscape, providing ubiquitous access to computational resources and services. Today, emerging technologies such as quantum computing, edge computing, and artificial intelligence promise to reshape the technological landscape once again, offering unprecedented capabilities for solving complex problems and processing vast amounts of data in real-time."
        
    #     "Climate change represents one of the most pressing challenges facing humanity in the 21st century, with far-reaching implications for ecosystems, economies, and societies worldwide. The scientific consensus indicates that human activities, particularly the burning of fossil fuels and deforestation, have significantly increased atmospheric concentrations of greenhouse gases, leading to global warming and altered weather patterns. The consequences include rising sea levels, more frequent and intense extreme weather events, disruptions to agricultural systems, and loss of biodiversity. Addressing climate change requires concerted efforts on multiple fronts, including transitioning to renewable energy sources, improving energy efficiency, developing carbon capture technologies, and implementing policies that reduce emissions. International cooperation and individual actions both play crucial roles in mitigating the impacts of climate change and adapting to its inevitable effects. The urgency of this challenge has spurred innovation in clean energy technologies and sustainable practices, offering hope for a more resilient and sustainable future.",
        
    #     "The field of medicine has undergone remarkable transformations throughout history, from ancient healing practices to modern evidence-based healthcare. The discovery of antibiotics revolutionized the treatment of bacterial infections, dramatically reducing mortality rates and extending human lifespan. Advances in medical imaging technologies, such as MRI, CT scans, and ultrasound, have enabled physicians to diagnose diseases with unprecedented accuracy and non-invasive methods. The development of vaccines has prevented countless deaths from infectious diseases and continues to be crucial in managing global health threats. Genetic engineering and biotechnology have opened new frontiers in treating genetic disorders and developing personalized medicine approaches. Telemedicine and digital health technologies have expanded access to healthcare services, particularly in remote and underserved areas. Despite these advances, challenges remain in addressing chronic diseases, healthcare disparities, and emerging health threats, requiring continued innovation and investment in medical research and healthcare delivery systems.",
        
    #     "Space exploration represents humanity's enduring curiosity about the cosmos and our quest to understand our place in the universe. From the early days of rocketry and the Space Race between superpowers to modern international collaboration on space stations and planetary missions, space exploration has driven technological innovation and expanded our understanding of the solar system and beyond. The Apollo missions to the Moon demonstrated human capability to travel beyond Earth, while robotic explorers have provided detailed insights into Mars, Jupiter, Saturn, and other celestial bodies. Space telescopes like Hubble and James Webb have revealed breathtaking images of distant galaxies and deepened our knowledge of cosmic evolution. Commercial space companies are now making space more accessible, developing reusable rockets and planning ambitious missions to Mars and beyond. The challenges of space travel, including radiation exposure, life support systems, and psychological effects on astronauts, continue to drive scientific research and technological development. As we look to the future, space exploration promises not only scientific discoveries but also potential solutions to Earth's challenges through resource utilization and new technologies.",
        
    #     "The evolution of the internet has fundamentally transformed how people communicate, work, and access information in the modern world. What began as a military research network has evolved into a global platform connecting billions of people and devices. The development of the World Wide Web made the internet accessible to non-technical users, sparking a revolution in information sharing and e-commerce. Social media platforms have changed how people interact and form communities, while streaming services have transformed entertainment consumption. Cloud computing has democratized access to powerful computing resources, enabling startups and individuals to build sophisticated applications without massive infrastructure investments. The internet of things (IoT) is connecting everyday devices to the network, creating smart homes and cities. However, this connectivity has also raised concerns about privacy, security, and digital divides. As technologies like 5G, edge computing, and artificial intelligence continue to evolve, the internet will likely undergo further transformations that we can barely imagine today.",
        
    #     "Economic systems and theories have evolved significantly throughout human history, reflecting changing social structures, technological capabilities, and philosophical perspectives. Traditional economies based on barter and subsistence gave way to market systems with the development of currency and trade routes. The Industrial Revolution transformed production methods and gave rise to capitalism, with its emphasis on private ownership and market competition. Socialist and communist theories emerged as alternatives, advocating for collective ownership and planned economies. Globalization has interconnected national economies, creating complex supply chains and international trade networks. Digital technologies have enabled new economic models, including platform economies and cryptocurrency systems. Economic challenges such as inequality, environmental sustainability, and financial stability continue to shape policy debates and economic thinking. Understanding economic principles and systems is crucial for addressing contemporary issues like resource allocation, wealth distribution, and sustainable development in an increasingly interconnected world.",
        
    #     "The study of human psychology encompasses the investigation of mental processes, behavior, and the factors that influence human thoughts and actions. From the early work of pioneers like Wilhelm Wundt and William James to modern neuroscience and cognitive psychology, the field has developed sophisticated methods for understanding the human mind. Key areas of study include perception, memory, learning, emotion, motivation, and social behavior. Psychological theories have practical applications in education, healthcare, business, and law enforcement. Mental health awareness has grown significantly, leading to better diagnosis and treatment of conditions like depression, anxiety, and schizophrenia. The nature versus nurture debate continues to shape research into how genetics and environment interact to influence human development. Modern psychology increasingly incorporates biological perspectives, using brain imaging and genetic studies to understand the neural basis of behavior. As our understanding of the mind grows, psychology continues to evolve, offering insights into human potential and wellbeing while addressing the challenges of mental health in an increasingly complex world.",
        
    #     "The development of renewable energy technologies represents one of the most significant technological transitions in modern history, driven by the need to address climate change and ensure sustainable energy supplies. Solar photovoltaic technology has advanced dramatically, with efficiency improvements and cost reductions making solar energy competitive with fossil fuels in many regions. Wind power has also matured, with both onshore and offshore installations contributing significantly to electricity generation in numerous countries. Energy storage technologies, particularly batteries, are evolving rapidly to address the intermittent nature of renewable sources. Hydrogen fuel cells and other emerging technologies offer promising alternatives for transportation and industrial applications. The transition to renewable energy requires substantial infrastructure investments, policy support, and technological innovation. Grid modernization and smart energy management systems are essential for integrating distributed renewable resources effectively. Despite challenges, the renewable energy sector continues to grow, offering pathways to reduce carbon emissions while meeting increasing global energy demands.",
        
    #     "Biotechnology has revolutionized numerous fields, from medicine and agriculture to environmental remediation and industrial manufacturing. Genetic engineering techniques like CRISPR have enabled precise modifications of DNA, opening new possibilities for treating genetic diseases and developing drought-resistant crops. Synthetic biology allows scientists to design and construct biological systems with specific functions, creating applications in biofuels, bioplastics, and pharmaceuticals. Biopharmaceuticals produced through biological processes offer targeted treatments for various diseases with fewer side effects than traditional chemical drugs. Agricultural biotechnology has increased crop yields, reduced pesticide use, and enhanced nutritional content of food crops. Environmental applications include bioremediation using microorganisms to clean up pollution and develop sustainable waste treatment processes. Industrial biotechnology enables the production of chemicals, materials, and fuels through biological processes, reducing reliance on petroleum-based feedstocks. As biotechnology continues to advance, ethical considerations and regulatory frameworks become increasingly important for ensuring responsible development and deployment of these powerful technologies.",
        
    #     "The field of robotics has evolved from simple automated machines to sophisticated systems capable of complex tasks and human-like interactions. Industrial robots have transformed manufacturing processes, performing repetitive tasks with precision and efficiency while improving workplace safety. Service robots are increasingly being deployed in healthcare, hospitality, and domestic environments, assisting with tasks ranging from surgery to elderly care. Autonomous robots, including self-driving vehicles and drones, are reshaping transportation and logistics industries. Humanoid robots and collaborative robots (cobots) are designed to work alongside humans, requiring advanced sensing, perception, and decision-making capabilities. Artificial intelligence and machine learning have enabled robots to adapt to new situations and learn from experience, making them more versatile and capable. The development of soft robotics and bio-inspired designs is creating robots with more natural movements and interactions. As robotics technology advances, questions about job displacement, safety, and ethical use become increasingly important for society to address.",
        
    #     "Quantum computing represents a paradigm shift in information processing, leveraging quantum mechanical phenomena to solve problems that are intractable for classical computers. Quantum bits, or qubits, can exist in superposition states, enabling quantum computers to explore multiple possibilities simultaneously. Quantum entanglement allows qubits to be correlated in ways that have no classical analog, providing additional computational power. Quantum algorithms like Shor's algorithm and Grover's algorithm offer exponential speedups for certain problems, potentially breaking current cryptographic systems and revolutionizing optimization tasks. Building practical quantum computers requires overcoming significant challenges in maintaining quantum coherence, error correction, and scaling up qubit numbers. Different approaches to quantum computing include superconducting circuits, trapped ions, photonic systems, and topological quantum computers, each with advantages and limitations. Near-term applications include quantum simulation for drug discovery and materials science, while long-term applications could transform cryptography, optimization, and artificial intelligence. The race for quantum advantage has spurred massive investment from governments, corporations, and research institutions worldwide.",
        
    #     "The globalization of financial markets has created an interconnected system where capital flows across borders with unprecedented speed and volume, presenting both opportunities and challenges for economic stability and development. International financial institutions like the IMF and World Bank play crucial roles in maintaining global financial stability and providing development financing. Cryptocurrencies and blockchain technologies are disrupting traditional financial systems, offering new possibilities for peer-to-peer transactions and decentralized finance. High-frequency trading and algorithmic trading have transformed market dynamics, raising questions about market fairness and systemic risk. Financial regulations and oversight mechanisms struggle to keep pace with technological innovations and global integration. Financial inclusion remains a significant challenge, with billions of people lacking access to basic financial services. Sustainable finance and impact investing are growing trends, directing capital toward environmentally and socially beneficial projects. The complexity and interconnectedness of global financial systems require sophisticated risk management and coordinated international policy responses to address potential crises and promote stable economic growth.",
        
    #     "Urbanization and smart city development are transforming how people live, work, and interact in metropolitan areas worldwide. The rapid growth of cities presents challenges in transportation, housing, energy consumption, and environmental sustainability. Smart city technologies leverage sensors, data analytics, and connectivity to optimize urban services and improve quality of life. Intelligent transportation systems use real-time data to manage traffic flow, reduce congestion, and enhance public transit efficiency. Smart grids enable more efficient energy distribution and integration of renewable energy sources. Urban planning increasingly incorporates sustainability principles, green spaces, and resilient infrastructure to address climate change and environmental concerns. Digital twins and simulation tools help city planners model and optimize urban systems before implementing changes. The Internet of Things (IoT) enables smart buildings, waste management systems, and environmental monitoring. As cities become more connected and data-driven, questions about privacy, digital equity, and governance become increasingly important for ensuring that smart city developments benefit all residents.",
        
    #     "The advancement of materials science has enabled breakthroughs across numerous industries, from aerospace and electronics to medicine and energy. Nanotechnology allows manipulation of materials at the atomic and molecular scale, creating materials with novel properties and applications. Advanced composites combine different materials to achieve superior strength-to-weight ratios, enabling lighter and more efficient aircraft, vehicles, and structures. Smart materials can respond to environmental stimuli like temperature, pressure, or light, finding applications in sensors, actuators, and adaptive structures. Biocompatible materials have revolutionized medical implants and drug delivery systems, improving patient outcomes and reducing rejection rates. Superconductors enable lossless electricity transmission and powerful electromagnets for medical imaging and particle accelerators. Sustainable materials development focuses on recyclable, biodegradable, and environmentally friendly alternatives to traditional materials. Computational materials science uses simulation and machine learning to accelerate the discovery and optimization of new materials. As materials science continues to advance, it enables technological innovations that address global challenges in energy, healthcare, transportation, and environmental sustainability."
    ]
    
 
    
    import time
    start_time = time.time()
    
    outputs = llm.generate(prompts, sampling_params)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate token throughput metrics
    total_prompt_tokens = sum(len(tokenizer.encode(p)) if isinstance(p, str) else len(p) for p in prompts)
    total_generated_tokens = sum(len(output['token_ids']) for output in outputs)
    total_tokens = total_prompt_tokens + total_generated_tokens
    
    print(f"\n{'='*60}")
    print(f"PERFORMANCE METRICS")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Number of sequences: {len(outputs)}")
    print(f"\nSequence throughput: {len(outputs)/total_time:.2f} seq/sec")
    print(f"Token throughput: {total_tokens/total_time:.2f} tokens/sec (input+output)")
    print(f"Generation throughput: {total_generated_tokens/total_time:.2f} tokens/sec (output only)")
    print(f"{'='*60}\n")
    
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        prompt_text = tokenizer.decode(prompt) if isinstance(prompt, list) else prompt
        print(f"\n{'='*80}")
        print(f"SEQUENCE {i+1}/{len(prompts)}")
        print(f"{'='*80}")
        print(f"Prompt: {prompt_text[:100]}...")
        print(f"\nCompletion ({len(output['text'])} chars):")
        print(output['text'][:500] + "..." if len(output['text']) > 500 else output['text'])
        print(f"\nTokens generated: {len(output['token_ids'])}")
        print(f"Prompt tokens: {len(tokenizer.encode(prompt)) if isinstance(prompt, str) else len(prompt)}")


if __name__ == "__main__":
    main()
