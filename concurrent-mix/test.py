import asyncio
import aiohttp
import json
import time
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import sys
import argparse

@dataclass
class RequestResult:
    request_id: int
    start_time: float
    end_time: float
    latency: float
    success: bool
    response_text: str
    error: str = None
    ttft: float = None  # Time to first token
    is_offline: bool = False  # Track if this was an offline request

class ConcurrentTester:
    def __init__(self,
                 base_url: str = "http://127.0.0.1:27712",
                 model: str = "Qwen2.5-7B-Instruct",
                 prompt: str = "William Hanna",
                 max_tokens: int = 10,
                 temperature: float = 0.0,
                 stream: bool = True,
                 offline_ratio: float = 0.5,  # Ratio of offline requests (0.0 to 1.0)
                 mixed_mode: bool = True,  # Enable mixed online/offline mode
                 seed: int = 42):  # Random seed for reproducibility
        self.base_url = base_url
        self.endpoint = f"{base_url}/v1/completions"
        self.model = model
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stream = stream
        self.offline_ratio = offline_ratio
        self.mixed_mode = mixed_mode
        self.seed = seed
        self.results: List[RequestResult] = []
        
    def get_request_body(self, is_offline: bool = False) -> Dict[str, Any]:
        return {
            "model": self.model,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": self.stream,
            "offline": is_offline
        }
    
    async def send_request(self, session: aiohttp.ClientSession, request_id: int, is_offline: bool = False) -> RequestResult:
        start_time = time.time()
        ttft = None
        response_text = ""

        try:
            request_body = self.get_request_body(is_offline)
            # print(f"Request {request_id} - Body: {json.dumps(request_body, indent=2)}")
            # print(f"Request {request_id} - URL: {self.endpoint}")

            async with session.post(
                self.endpoint,
                json=request_body,
                headers={"Content-Type": "application/json"}
            ) as response:
                if self.stream:
                    first_token_received = False
                    async for line in response.content:
                        if not first_token_received:
                            ttft = time.time() - start_time
                            first_token_received = True
                        
                        line = line.decode('utf-8').strip()
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    if "text" in chunk["choices"][0]:
                                        response_text += chunk["choices"][0]["text"]
                            except json.JSONDecodeError:
                                pass
                else:
                    text = await response.text()
                    response_data = json.loads(text)
                    if "choices" in response_data and len(response_data["choices"]) > 0:
                        response_text = response_data["choices"][0]["text"]
                
                end_time = time.time()
                return RequestResult(
                    request_id=request_id,
                    start_time=start_time,
                    end_time=end_time,
                    latency=end_time - start_time,
                    success=True,
                    response_text=response_text,
                    ttft=ttft,
                    is_offline=is_offline
                )
                
        except Exception as e:
            end_time = time.time()
            return RequestResult(
                request_id=request_id,
                start_time=start_time,
                end_time=end_time,
                latency=end_time - start_time,
                success=False,
                response_text="",
                error=str(e),
                is_offline=is_offline
            )
    
    async def run_concurrent_test(self,
                                  total_requests: int = 1000,
                                  requests_per_second: float = 5.0) -> None:
        print(f"Starting concurrent test with {total_requests} requests at ~{requests_per_second} req/s")
        print(f"Model: {self.model}")
        print(f"Endpoint: {self.endpoint}")
        print(f"Stream: {self.stream}")
        if self.mixed_mode:
            offline_count = int(total_requests * self.offline_ratio)
            online_count = total_requests - offline_count
            print(f"Mixed Mode: {online_count} online, {offline_count} offline (ratio: {self.offline_ratio:.1%})")
        print("-" * 50)
        
        connector = aiohttp.TCPConnector(limit=100)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            test_start_time = time.time()
            
            interval = 1.0 / requests_per_second
            
            # Determine which requests should be offline
            import random
            random.seed(self.seed)  # Use configurable seed for reproducibility
            request_types = []
            if self.mixed_mode:
                offline_count = int(total_requests * self.offline_ratio)
                request_types = [True] * offline_count + [False] * (total_requests - offline_count)
                random.shuffle(request_types)
            else:
                request_types = [False] * total_requests
            
            print(f"Number of online requests: {sum(request_types)}")

            for i in range(total_requests):
                is_offline = request_types[i]
                task = asyncio.create_task(self.send_request(session, i, is_offline))
                tasks.append(task)
                
                if (i + 1) % 10 == 0:
                    print(f"Queued {i + 1}/{total_requests} requests...")
                
                if i < total_requests - 1:
                    await asyncio.sleep(interval)
            
            print(f"\nAll requests queued. Waiting for responses...")
            results = await asyncio.gather(*tasks)
            self.results = results
            
            test_end_time = time.time()
            total_test_time = test_end_time - test_start_time
            
            self.print_results(total_test_time)
    
    def print_results(self, total_test_time: float) -> None:
        successful_requests = [r for r in self.results if r.success]
        failed_requests = [r for r in self.results if not r.success]

        # Separate online and offline requests
        online_requests = [r for r in self.results if not r.is_offline]
        offline_requests = [r for r in self.results if r.is_offline]
        online_successful = [r for r in successful_requests if not r.is_offline]
        offline_successful = [r for r in successful_requests if r.is_offline]

        print("\n" + "=" * 50)
        print("TEST RESULTS")
        print("=" * 50)

        print(f"Total requests: {len(self.results)}")
        print(f"  Online: {len(online_requests)} (successful: {len(online_successful)})")
        print(f"  Offline: {len(offline_requests)} (successful: {len(offline_successful)})")
        print(f"Successful: {len(successful_requests)}")
        print(f"Failed: {len(failed_requests)}")
        print(f"Success rate: {len(successful_requests)/len(self.results)*100:.2f}%")
        print(f"Total test time: {total_test_time:.2f} seconds")
        print(f"Actual request rate: {len(self.results)/total_test_time:.2f} req/s")
        
        if successful_requests:
            latencies = [r.latency for r in successful_requests]
            latencies.sort()

            print("\nLatency Statistics (all successful requests):")
            print(f"  Min: {min(latencies):.3f}s")
            print(f"  Max: {max(latencies):.3f}s")
            print(f"  Mean: {sum(latencies)/len(latencies):.3f}s")
            print(f"  Median: {latencies[len(latencies)//2]:.3f}s")
            print(f"  P90: {latencies[int(len(latencies)*0.9)]:.3f}s")
            print(f"  P95: {latencies[int(len(latencies)*0.95)]:.3f}s")
            print(f"  P99: {latencies[int(len(latencies)*0.99)]:.3f}s")

            # Online latency statistics
            if online_successful:
                online_latencies = [r.latency for r in online_successful]
                online_latencies.sort()
                print("\nOnline Request Latency Statistics:")
                print(f"  Min: {min(online_latencies):.3f}s")
                print(f"  Max: {max(online_latencies):.3f}s")
                print(f"  Mean: {sum(online_latencies)/len(online_latencies):.3f}s")
                print(f"  Median: {online_latencies[len(online_latencies)//2]:.3f}s")

            # Offline latency statistics
            if offline_successful:
                offline_latencies = [r.latency for r in offline_successful]
                offline_latencies.sort()
                print("\nOffline Request Latency Statistics:")
                print(f"  Min: {min(offline_latencies):.3f}s")
                print(f"  Max: {max(offline_latencies):.3f}s")
                print(f"  Mean: {sum(offline_latencies)/len(offline_latencies):.3f}s")
                print(f"  Median: {offline_latencies[len(offline_latencies)//2]:.3f}s")
            
            if self.stream:
                ttfts = [r.ttft for r in successful_requests if r.ttft is not None]
                if ttfts:
                    ttfts.sort()
                    print("\nTime to First Token (TTFT) Statistics:")
                    print(f"  Min: {min(ttfts):.3f}s")
                    print(f"  Max: {max(ttfts):.3f}s")
                    print(f"  Mean: {sum(ttfts)/len(ttfts):.3f}s")
                    print(f"  Median: {ttfts[len(ttfts)//2]:.3f}s")
        
        if failed_requests:
            print(f"\nFailed Requests ({len(failed_requests)}):")
            for i, r in enumerate(failed_requests[:5]):
                print(f"  Request {r.request_id}: {r.error}")
            if len(failed_requests) > 5:
                print(f"  ... and {len(failed_requests) - 5} more")
        
        print("\nSample Responses (first 3):")
        for i, r in enumerate(successful_requests[:3]):
            print(f"  Request {r.request_id}: {repr(r.response_text)}")
    
    def save_results(self, filename: str = None) -> None:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dirname = f"local/log/concurrent_test"
            filename = f"{dirname}/{timestamp}.json"
        
        results_data = []
        for r in self.results:
            results_data.append({
                "request_id": r.request_id,
                "start_time": r.start_time,
                "end_time": r.end_time,
                "latency": r.latency,
                "success": r.success,
                "response_text": r.response_text,
                "error": r.error,
                "ttft": r.ttft,
                "is_offline": r.is_offline
            })
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nResults saved to {filename}")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Concurrent testing tool for LLM inference endpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Request parameters
    parser.add_argument(
        "--total-requests", "-n",
        type=int,
        default=100,
        help="Total number of requests to send"
    )
    parser.add_argument(
        "--requests-per-second", "-r",
        type=float,
        default=5.0,
        help="Target request rate (requests per second)"
    )

    # Server configuration
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:27712",
        help="Base URL of the inference server"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen2.5-7B-Instruct",
        help="Model name to use for inference"
    )

    # Prompt configuration
    parser.add_argument(
        "--prompt",
        type=str,
        default="William Hanna",
        help="Input prompt for the model"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling"
    )

    # Request options
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming mode"
    )

    # Mixed mode options
    parser.add_argument(
        "--offline-ratio",
        type=float,
        default=0.5,
        help="Ratio of offline requests (0.0 to 1.0, default 0.5 = 50%% offline)"
    )
    parser.add_argument(
        "--no-mixed-mode",
        action="store_true",
        help="Disable mixed mode (all requests will be online)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible request distribution"
    )

    # Output options
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save results to a JSON file"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output filename for results (auto-generated if not specified)"
    )

    return parser.parse_args()

async def main():
    args = parse_arguments()

    tester = ConcurrentTester(
        base_url=args.base_url,
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stream=not args.no_stream,
        offline_ratio=args.offline_ratio,
        mixed_mode=not args.no_mixed_mode,
        seed=args.seed
    )

    try:
        await tester.run_concurrent_test(
            total_requests=args.total_requests,
            requests_per_second=args.requests_per_second
        )

        if args.save_results:
            tester.save_results(args.output_file)

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())