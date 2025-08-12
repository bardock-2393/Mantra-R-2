#!/usr/bin/env python3
"""
Test Redis connection
"""

import redis
from config import Config

def test_redis():
    """Test Redis connection"""
    print("🧪 Testing Redis connection...")
    
    try:
        # Test Redis connection
        redis_client = redis.from_url(Config.REDIS_URL)
        
        # Test basic operations
        test_key = "test:connection"
        test_value = "Hello Redis!"
        
        # Set value
        redis_client.set(test_key, test_value)
        print(f"✅ Set test value: {test_key} = {test_value}")
        
        # Get value
        retrieved_value = redis_client.get(test_key)
        if retrieved_value:
            retrieved_value = retrieved_value.decode('utf-8')
            print(f"✅ Retrieved test value: {retrieved_value}")
        
        # Test hash operations
        test_hash_key = "test:session:123"
        test_data = {
            'user_id': '123',
            'timestamp': '2024-01-15T10:30:00',
            'status': 'active'
        }
        
        # Set hash
        redis_client.hset(test_hash_key, mapping=test_data)
        print(f"✅ Set test hash: {test_hash_key}")
        
        # Get hash
        retrieved_hash = redis_client.hgetall(test_hash_key)
        print(f"✅ Retrieved test hash: {retrieved_hash}")
        
        # Test expiration
        redis_client.expire(test_hash_key, 60)  # 60 seconds
        ttl = redis_client.ttl(test_hash_key)
        print(f"✅ Set expiration: {ttl} seconds")
        
        # Clean up test data
        redis_client.delete(test_key)
        redis_client.delete(test_hash_key)
        print("✅ Cleaned up test data")
        
        print("🎉 Redis connection test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Redis connection test failed: {e}")
        return False

if __name__ == "__main__":
    test_redis()
