from github import Github
import slack
from typing import Dict, List, Optional, Any
import logging
from src.core.config.configurations import PRValidationConfig
import subprocess
import requests
import tempfile

class PRValidator:
    """Validate and manage pull requests"""
    def __init__(self, config: PRValidationConfig):
        self.config = config
        
        # Initialize clients
        self.github = Github(config.github_token)
        self.repo = self.github.get_repo(config.repository)
        self.slack_client = slack.WebClient(token=config.notification_channels['slack'])
        
        # Setup logging
        self.logger = logging.getLogger("pr_validator")
        self._setup_logging()
        
    def validate_pull_request(self, pr_number: int) -> bool:
        """Run all validation checks on PR"""
        pr = self.repo.get_pull(pr_number)
        
        validation_results = {
            'size_check': self._check_pr_size(pr),
            'tests_check': self._check_required_tests(pr),
            'coverage_check': self._check_coverage(pr),
            'style_check': self._check_code_style(pr),
            'security_check': self._check_security(pr),
            'review_check': self._check_reviews(pr)
        }
        
        # Send notifications
        self._send_validation_report(pr, validation_results)
        
        # Update PR status
        self._update_pr_status(pr, validation_results)
        
        return all(validation_results.values())
        
    def _check_pr_size(self, pr) -> bool:
        """Check PR size limits"""
        changes = pr.additions + pr.deletions
        
        if changes > self.config.size_limits['total']:
            self._add_pr_comment(
                pr,
                f"❌ PR exceeds size limit: {changes} changes > "
                f"{self.config.size_limits['total']} limit"
            )
            return False
            
        files_changed = pr.changed_files
        if files_changed > self.config.size_limits['files']:
            self._add_pr_comment(
                pr,
                f"❌ Too many files changed: {files_changed} > "
                f"{self.config.size_limits['files']} limit"
            )
            return False
            
        return True
        
    def _check_required_tests(self, pr) -> bool:
        """Check required test status"""
        commit = pr.get_commits().reversed[0]  # Get latest commit
        statuses = commit.get_combined_status()
        
        missing_checks = set(self.config.required_checks) - {
            status.context for status in statuses.statuses
        }
        
        if missing_checks:
            self._add_pr_comment(
                pr,
                f"❌ Missing required checks: {', '.join(missing_checks)}"
            )
            return False
            
        failed_checks = [
            status.context
            for status in statuses.statuses
            if status.state != 'success'
        ]
        
        if failed_checks:
            self._add_pr_comment(
                pr,
                f"❌ Failed checks: {', '.join(failed_checks)}"
            )
            return False
            
        return True
        
    def _check_coverage(self, pr) -> bool:
        """Check code coverage requirements"""
        try:
            coverage_report = self._get_coverage_report(pr)
            
            if coverage_report['total'] < 80:  # Minimum coverage threshold
                self._add_pr_comment(
                    pr,
                    f"❌ Insufficient code coverage: {coverage_report['total']}% < 80%\n"
                    f"Please add tests to improve coverage."
                )
                return False
                
            # Check coverage diff
            if coverage_report['diff'] < 0:
                self._add_pr_comment(
                    pr,
                    f"❌ Coverage decrease: {coverage_report['diff']}%\n"
                    f"Please maintain or improve code coverage."
                )
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Coverage check error: {str(e)}")
            return False
            
    def _check_code_style(self, pr) -> bool:
        """Check code style requirements"""
        try:
            # Run linting
            lint_results = self._run_linting(pr)
            
            if lint_results['errors']:
                self._add_pr_comment(
                    pr,
                    "❌ Style check failed:\n" +
                    "\n".join(f"- {error}" for error in lint_results['errors'])
                )
                return False
                
            if lint_results['warnings']:
                self._add_pr_comment(
                    pr,
                    "⚠️ Style warnings:\n" +
                    "\n".join(f"- {warning}" for warning in lint_results['warnings'])
                )
                
            return not lint_results['errors']
            
        except Exception as e:
            self.logger.error(f"Style check error: {str(e)}")
            return False
            
    def _check_security(self, pr) -> bool:
        """Check security requirements"""
        try:
            # Run security scan
            security_results = self._run_security_scan(pr)
            
            if security_results['vulnerabilities']:
                self._add_pr_comment(
                    pr,
                    "❌ Security vulnerabilities found:\n" +
                    "\n".join(
                        f"- {vuln['severity']} - {vuln['description']}"
                        for vuln in security_results['vulnerabilities']
                    )
                )
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Security check error: {str(e)}")
            return False
            
    def _check_reviews(self, pr) -> bool:
        """Check review requirements"""
        reviews = pr.get_reviews()
        
        # Count approvals
        approvals = sum(1 for review in reviews if review.state == 'APPROVED')
        
        if approvals < self.config.required_reviewers:
            self._add_pr_comment(
                pr,
                f"❌ Need {self.config.required_reviewers - approvals} more approval(s)"
            )
            return False
            
        # Check for blocking reviews
        if any(review.state == 'CHANGES_REQUESTED' for review in reviews):
            self._add_pr_comment(
                pr,
                "❌ Changes requested by reviewers must be addressed"
            )
            return False
            
        return True
        
    def _send_validation_report(self, pr, results: Dict[str, bool]):
        """Send validation report to notification channels"""
        # Create report message
        status = "✅ Passed" if all(results.values()) else "❌ Failed"
        message = f"PR #{pr.number} Validation {status}\n\n"
        
        for check, passed in results.items():
            icon = "✅" if passed else "❌"
            message += f"{icon} {check}\n"
            
        # Send to Slack
        if 'slack' in self.config.notification_channels:
            self.slack_client.chat_postMessage(
                channel=self.config.notification_channels['slack'],
                text=message,
                blocks=[
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": message}
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"<{pr.html_url}|View Pull Request>"
                        }
                    }
                ]
            )
            
        # Send to email if configured
        if 'email' in self.config.notification_channels:
            self._send_email_report(
                self.config.notification_channels['email'],
                f"PR #{pr.number} Validation Report",
                message
            )
            
    def _update_pr_status(self, pr, results: Dict[str, bool]):
        """Update PR status checks"""
        commit = pr.get_commits().reversed[0]  # Get latest commit
        
        for check, passed in results.items():
            state = 'success' if passed else 'failure'
            commit.create_status(
                state=state,
                target_url=pr.html_url,
                description=f"{check} validation {'passed' if passed else 'failed'}",
                context=f"validation/{check}"
            )
            
    def _add_pr_comment(self, pr, message: str):
        """Add comment to PR"""
        pr.create_issue_comment(message)
        
    def _get_coverage_report(self, pr) -> Dict[str, float]:
        """Get code coverage report"""
        # Implementation depends on coverage tool
        # This is a placeholder
        return {
            'total': 85.0,
            'diff': 0.5
        }
        
    def _run_linting(self, pr) -> Dict[str, List[str]]:
        """Run code linting"""
        errors = []
        warnings = []
        
        # Get changed files
        files = pr.get_files()
        
        for file in files:
            if file.filename.endswith('.py'):
                # Run pylint
                result = self._run_pylint(file.raw_url)
                errors.extend(result['errors'])
                warnings.extend(result['warnings'])
                
        return {
            'errors': errors,
            'warnings': warnings
        }
        
    def _run_security_scan(self, pr) -> Dict[str, List[Dict[str, str]]]:
        """Run security vulnerability scan"""
        vulnerabilities = []
        
        # Get dependencies
        requirements = self._get_requirements(pr)
        
        # Check each dependency
        for dependency in requirements:
            # Check known vulnerabilities database
            vulns = self._check_vulnerability_database(dependency)
            vulnerabilities.extend(vulns)
            
        return {
            'vulnerabilities': vulnerabilities
        }
        
    def _run_pylint(self, file_url: str) -> Dict[str, List[str]]:
        """Run pylint on file"""
        # Download file content
        response = requests.get(file_url)
        content = response.text
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(suffix='.py') as temp:
            temp.write(content.encode())
            temp.flush()
            
            # Run pylint
            result = subprocess.run(
                ['pylint', temp.name],
                capture_output=True,
                text=True
            )
            
        # Parse output
        errors = []
        warnings = []
        
        for line in result.stdout.splitlines():
            if 'error' in line.lower():
                errors.append(line)
            elif 'warning' in line.lower():
                warnings.append(line)
                
        return {
            'errors': errors,
            'warnings': warnings
        }
        
    def _get_requirements(self, pr) -> List[str]:
        """Get project dependencies"""
        try:
            # Try to find requirements.txt
            contents = pr.get_files()
            for content in contents:
                if content.filename == 'requirements.txt':
                    return content.decoded_content.decode().splitlines()
                    
        except Exception as e:
            self.logger.error(f"Error getting requirements: {str(e)}")
            
        return []
        
    def _check_vulnerability_database(self, dependency: str) -> List[Dict[str, str]]:
        """Check dependency for known vulnerabilities"""
        # This would typically use a security advisory database API
        # This is a placeholder implementation
        return []
        
    def _send_email_report(self, recipient: str, subject: str, body: str):
        """Send email report"""
        # Implementation depends on email service
        # This is a placeholder
        pass
        
class AutoMerger:
    """Automatic PR merger"""
    def __init__(self, config: PRValidationConfig):
        self.config = config
        self.github = Github(config.github_token)
        self.repo = self.github.get_repo(config.repository)
        
    def check_auto_merge(self, pr_number: int):
        """Check if PR can be auto-merged"""
        pr = self.repo.get_pull(pr_number)
        
        # Check if PR has auto-merge label
        if not any(label.name in self.config.auto_merge_labels 
                  for label in pr.labels):
            return
            
        # Validate PR
        validator = PRValidator(self.config)
        if not validator.validate_pull_request(pr_number):
            return
            
        # Check if PR is mergeable
        if not pr.mergeable:
            pr.create_issue_comment(
                "❌ Cannot auto-merge: merge conflicts need to be resolved"
            )
            return
            
        # Merge PR
        try:
            pr.merge(
                merge_method='squash',
                commit_title=f"Auto-merge PR #{pr_number}: {pr.title}"
            )
        except Exception as e:
            pr.create_issue_comment(
                f"❌ Auto-merge failed: {str(e)}"
            )

if __name__ == "__main__":
    # Example usage
    config = PRValidationConfig(
        github_token="your-token",
        repository="owner/repo",
        required_checks=['tests', 'coverage', 'style', 'security'],
        size_limits={'total': 1000, 'files': 50},
        notification_channels={
            'slack': 'channel-id',
            'email': 'team@example.com'
        },
        auto_merge_labels=['auto-merge'],
        required_reviewers=2
    )
    
    validator = PRValidator(config)
    validator.validate_pull_request(123)  # PR number
    
    auto_merger = AutoMerger(config)
    auto_merger.check_auto_merge(123)  # PR number