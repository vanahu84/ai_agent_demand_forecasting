"""
Git Integration Module for MLOps Pipeline.

This module provides Git version control integration for automated pipeline triggers,
code versioning, and artifact tracking with commit information.
"""

import asyncio
import json
import logging
import os
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GitCommit:
    """Git commit information."""
    hash: str
    author: str
    email: str
    timestamp: datetime
    message: str
    files_changed: List[str]
    branch: str


@dataclass
class GitWebhookPayload:
    """Git webhook payload structure."""
    repository: str
    branch: str
    commits: List[GitCommit]
    pusher: str
    timestamp: datetime


class GitIntegration:
    """Git integration for MLOps pipeline automation."""
    
    def __init__(self, repository_path: Optional[str] = None):
        """Initialize Git integration."""
        self.repository_path = repository_path or os.getcwd()
        self.webhook_handlers: Dict[str, callable] = {}
        self.branch_policies: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default branch policies
        self._initialize_default_policies()
    
    def _initialize_default_policies(self):
        """Initialize default branch policies."""
        self.branch_policies = {
            "main": {
                "auto_trigger_pipeline": True,
                "require_tests": True,
                "require_approval": False,
                "model_types": ["all"]
            },
            "develop": {
                "auto_trigger_pipeline": True,
                "require_tests": True,
                "require_approval": False,
                "model_types": ["all"]
            },
            "feature/*": {
                "auto_trigger_pipeline": False,
                "require_tests": True,
                "require_approval": True,
                "model_types": []
            }
        }
    
    def is_git_repository(self) -> bool:
        """Check if the current directory is a Git repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.repository_path,
                capture_output=True,
                text=True,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def get_current_commit(self) -> Optional[GitCommit]:
        """Get current Git commit information."""
        try:
            if not self.is_git_repository():
                logger.warning("Not a Git repository")
                return None
            
            # Get commit hash
            hash_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repository_path,
                capture_output=True,
                text=True,
                check=True
            )
            commit_hash = hash_result.stdout.strip()
            
            # Get commit info
            info_result = subprocess.run(
                ["git", "show", "--format=%an|%ae|%ct|%s", "--name-only", commit_hash],
                cwd=self.repository_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            lines = info_result.stdout.strip().split('\n')
            if len(lines) < 2:
                return None
            
            # Parse commit info
            info_parts = lines[0].split('|')
            if len(info_parts) < 4:
                return None
            
            author = info_parts[0]
            email = info_parts[1]
            timestamp = datetime.fromtimestamp(int(info_parts[2]))
            message = info_parts[3]
            
            # Get changed files (skip empty lines)
            files_changed = [line for line in lines[1:] if line.strip()]
            
            # Get current branch
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.repository_path,
                capture_output=True,
                text=True,
                check=True
            )
            branch = branch_result.stdout.strip()
            
            return GitCommit(
                hash=commit_hash,
                author=author,
                email=email,
                timestamp=timestamp,
                message=message,
                files_changed=files_changed,
                branch=branch
            )
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting Git commit info: {str(e)}")
            return None
    
    def get_commits_since(self, since_hash: str) -> List[GitCommit]:
        """Get commits since a specific hash."""
        try:
            if not self.is_git_repository():
                return []
            
            # Get commit hashes since the specified hash
            result = subprocess.run(
                ["git", "rev-list", f"{since_hash}..HEAD"],
                cwd=self.repository_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            commit_hashes = result.stdout.strip().split('\n')
            if not commit_hashes or commit_hashes == ['']:
                return []
            
            commits = []
            for commit_hash in commit_hashes:
                commit = self._get_commit_info(commit_hash.strip())
                if commit:
                    commits.append(commit)
            
            return commits
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Error getting commits since {since_hash}: {str(e)}")
            return []
    
    def _get_commit_info(self, commit_hash: str) -> Optional[GitCommit]:
        """Get information for a specific commit."""
        try:
            # Get commit info
            info_result = subprocess.run(
                ["git", "show", "--format=%an|%ae|%ct|%s", "--name-only", commit_hash],
                cwd=self.repository_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            lines = info_result.stdout.strip().split('\n')
            if len(lines) < 2:
                return None
            
            # Parse commit info
            info_parts = lines[0].split('|')
            if len(info_parts) < 4:
                return None
            
            author = info_parts[0]
            email = info_parts[1]
            timestamp = datetime.fromtimestamp(int(info_parts[2]))
            message = info_parts[3]
            
            # Get changed files
            files_changed = [line for line in lines[1:] if line.strip()]
            
            # Get branch for this commit
            branch_result = subprocess.run(
                ["git", "branch", "--contains", commit_hash],
                cwd=self.repository_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse branch output (may contain multiple branches)
            branch_lines = branch_result.stdout.strip().split('\n')
            branch = "unknown"
            for line in branch_lines:
                if line.startswith('*'):
                    branch = line[2:].strip()
                    break
                elif line.strip():
                    branch = line.strip()
                    break
            
            return GitCommit(
                hash=commit_hash,
                author=author,
                email=email,
                timestamp=timestamp,
                message=message,
                files_changed=files_changed,
                branch=branch
            )
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed for commit {commit_hash}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting commit info for {commit_hash}: {str(e)}")
            return None
    
    def should_trigger_pipeline(self, commit: GitCommit) -> Tuple[bool, List[str]]:
        """Determine if a commit should trigger pipeline and which model types."""
        try:
            # Check branch policies
            policy = self._get_branch_policy(commit.branch)
            
            if not policy.get("auto_trigger_pipeline", False):
                return False, []
            
            # Check if any relevant files were changed
            relevant_files = self._get_relevant_files(commit.files_changed)
            if not relevant_files:
                return False, []
            
            # Determine model types to trigger
            model_types = policy.get("model_types", [])
            if "all" in model_types:
                model_types = self._infer_model_types_from_files(relevant_files)
            
            logger.info(f"Commit {commit.hash[:8]} should trigger pipeline for models: {model_types}")
            return True, model_types
            
        except Exception as e:
            logger.error(f"Error evaluating pipeline trigger for commit {commit.hash}: {str(e)}")
            return False, []
    
    def _get_branch_policy(self, branch: str) -> Dict[str, Any]:
        """Get policy for a specific branch."""
        # Direct match
        if branch in self.branch_policies:
            return self.branch_policies[branch]
        
        # Pattern match
        for pattern, policy in self.branch_policies.items():
            if '*' in pattern:
                prefix = pattern.replace('*', '')
                if branch.startswith(prefix):
                    return policy
        
        # Default policy
        return {
            "auto_trigger_pipeline": False,
            "require_tests": True,
            "require_approval": True,
            "model_types": []
        }
    
    def _get_relevant_files(self, files_changed: List[str]) -> List[str]:
        """Filter files that are relevant for ML pipeline."""
        relevant_patterns = [
            # Python files
            '.py',
            # Data files
            '.csv', '.json', '.parquet', '.pkl',
            # Configuration files
            '.yaml', '.yml', '.toml', '.ini',
            # Model files
            '.joblib', '.h5', '.onnx',
            # Notebook files
            '.ipynb',
            # Requirements
            'requirements.txt', 'pyproject.toml', 'setup.py'
        ]
        
        relevant_files = []
        for file_path in files_changed:
            if any(file_path.endswith(pattern) for pattern in relevant_patterns):
                relevant_files.append(file_path)
        
        return relevant_files
    
    def _infer_model_types_from_files(self, files_changed: List[str]) -> List[str]:
        """Infer model types from changed files."""
        model_types = set()
        
        # Check for specific model type indicators in file paths
        for file_path in files_changed:
            file_lower = file_path.lower()
            
            if 'xgboost' in file_lower or 'xgb' in file_lower:
                model_types.add('xgboost')
            elif 'prophet' in file_lower:
                model_types.add('prophet')
            elif 'arima' in file_lower:
                model_types.add('arima')
            elif 'ensemble' in file_lower:
                model_types.add('ensemble')
            elif any(keyword in file_lower for keyword in ['model', 'forecast', 'predict']):
                # Generic model files - trigger all model types
                model_types.update(['xgboost', 'prophet', 'arima', 'ensemble'])
        
        # If no specific model types found, default to xgboost
        if not model_types:
            model_types.add('xgboost')
        
        return list(model_types)
    
    async def handle_webhook(self, payload: Dict[str, Any]) -> List[str]:
        """Handle Git webhook payload and trigger pipelines if needed."""
        try:
            # Parse webhook payload
            webhook_payload = self._parse_webhook_payload(payload)
            if not webhook_payload:
                return []
            
            triggered_runs = []
            
            # Process each commit
            for commit in webhook_payload.commits:
                should_trigger, model_types = self.should_trigger_pipeline(commit)
                
                if should_trigger:
                    # Trigger pipeline for each model type
                    for model_type in model_types:
                        config = {
                            "git_commit": commit.hash,
                            "git_branch": commit.branch,
                            "git_author": commit.author,
                            "trigger_source": "git_webhook",
                            "hyperparameters": self._get_default_hyperparameters(model_type)
                        }
                        
                        # This would be called by the CI/CD integration
                        logger.info(f"Would trigger {model_type} pipeline for commit {commit.hash[:8]}")
                        triggered_runs.append(f"{model_type}_{commit.hash[:8]}")
            
            return triggered_runs
            
        except Exception as e:
            logger.error(f"Error handling webhook: {str(e)}")
            return []
    
    def _parse_webhook_payload(self, payload: Dict[str, Any]) -> Optional[GitWebhookPayload]:
        """Parse webhook payload into structured format."""
        try:
            # Support GitHub webhook format
            if 'repository' in payload and 'commits' in payload:
                repository = payload['repository'].get('full_name', 'unknown')
                branch = payload.get('ref', '').replace('refs/heads/', '')
                pusher = payload.get('pusher', {}).get('name', 'unknown')
                
                commits = []
                for commit_data in payload.get('commits', []):
                    commit = GitCommit(
                        hash=commit_data.get('id', ''),
                        author=commit_data.get('author', {}).get('name', ''),
                        email=commit_data.get('author', {}).get('email', ''),
                        timestamp=datetime.fromisoformat(
                            commit_data.get('timestamp', '').replace('Z', '+00:00')
                        ),
                        message=commit_data.get('message', ''),
                        files_changed=commit_data.get('added', []) + 
                                     commit_data.get('modified', []) + 
                                     commit_data.get('removed', []),
                        branch=branch
                    )
                    commits.append(commit)
                
                return GitWebhookPayload(
                    repository=repository,
                    branch=branch,
                    commits=commits,
                    pusher=pusher,
                    timestamp=datetime.now()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing webhook payload: {str(e)}")
            return None
    
    def _get_default_hyperparameters(self, model_type: str) -> Dict[str, Any]:
        """Get default hyperparameters for a model type."""
        defaults = {
            "xgboost": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8
            },
            "prophet": {
                "seasonality_mode": "multiplicative",
                "yearly_seasonality": True,
                "weekly_seasonality": True,
                "daily_seasonality": False
            },
            "arima": {
                "order": [1, 1, 1],
                "seasonal_order": [1, 1, 1, 12]
            },
            "ensemble": {
                "base_models": ["xgboost", "prophet", "arima"],
                "voting": "soft",
                "weights": [0.4, 0.4, 0.2]
            }
        }
        
        return defaults.get(model_type, {})
    
    def tag_model_version(self, model_id: str, version: str, commit_hash: Optional[str] = None) -> bool:
        """Tag a model version in Git."""
        try:
            if not self.is_git_repository():
                logger.warning("Not a Git repository - cannot create tag")
                return False
            
            if not commit_hash:
                current_commit = self.get_current_commit()
                if not current_commit:
                    logger.error("Cannot get current commit for tagging")
                    return False
                commit_hash = current_commit.hash
            
            tag_name = f"model-{model_id}-v{version}"
            tag_message = f"Model {model_id} version {version} at commit {commit_hash[:8]}"
            
            # Create annotated tag
            result = subprocess.run(
                ["git", "tag", "-a", tag_name, "-m", tag_message, commit_hash],
                cwd=self.repository_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Created Git tag: {tag_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create Git tag: {e}")
            return False
        except Exception as e:
            logger.error(f"Error creating Git tag: {str(e)}")
            return False
    
    def get_model_tags(self) -> List[Dict[str, str]]:
        """Get all model-related Git tags."""
        try:
            if not self.is_git_repository():
                return []
            
            # Get all tags with model prefix
            result = subprocess.run(
                ["git", "tag", "-l", "model-*"],
                cwd=self.repository_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            tags = []
            for tag_name in result.stdout.strip().split('\n'):
                if tag_name.strip():
                    # Parse tag name: model-{model_id}-v{version}
                    parts = tag_name.split('-')
                    if len(parts) >= 3:
                        model_id = '-'.join(parts[1:-1])  # Handle model IDs with dashes
                        version = parts[-1].replace('v', '')
                        
                        tags.append({
                            'tag_name': tag_name,
                            'model_id': model_id,
                            'version': version
                        })
            
            return tags
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get Git tags: {e}")
            return []
        except Exception as e:
            logger.error(f"Error getting Git tags: {str(e)}")
            return []
    
    def update_branch_policy(self, branch: str, policy: Dict[str, Any]):
        """Update branch policy configuration."""
        self.branch_policies[branch] = policy
        logger.info(f"Updated branch policy for {branch}: {policy}")
    
    def get_repository_info(self) -> Dict[str, Any]:
        """Get repository information."""
        try:
            if not self.is_git_repository():
                return {"is_git_repo": False}
            
            # Get remote URL
            remote_result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=self.repository_path,
                capture_output=True,
                text=True
            )
            remote_url = remote_result.stdout.strip() if remote_result.returncode == 0 else "unknown"
            
            # Get current branch
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.repository_path,
                capture_output=True,
                text=True,
                check=True
            )
            current_branch = branch_result.stdout.strip()
            
            # Get current commit
            current_commit = self.get_current_commit()
            
            return {
                "is_git_repo": True,
                "remote_url": remote_url,
                "current_branch": current_branch,
                "current_commit": current_commit.hash if current_commit else None,
                "repository_path": self.repository_path,
                "branch_policies": self.branch_policies
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {e}")
            return {"is_git_repo": True, "error": str(e)}
        except Exception as e:
            logger.error(f"Error getting repository info: {str(e)}")
            return {"is_git_repo": False, "error": str(e)}


# Global Git integration instance
git_integration = GitIntegration()