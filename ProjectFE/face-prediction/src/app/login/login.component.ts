import { Component, OnInit } from '@angular/core';
import { LoginService } from './login.service';
import { FormControl, Validators } from '@angular/forms';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css'],
})
export class LoginComponent implements OnInit {
  error: string;
  email = new FormControl('', [Validators.required, Validators.email]);
  password = new FormControl('', [Validators.required]);

  constructor(private loginService: LoginService) {
    this.error = '';
  }

  ngOnInit(): void {}

  submit(): void {
    if (this.email.value != null && this.password.value != null)
      this.loginService
        .sendCredentials(this.email.value, this.password.value)
        .subscribe((data) => {
          console.log(data);
        });
  }

  getEmailErrorMessage() {
    if (this.email.hasError('required')) {
      return 'You must enter a value';
    }

    return this.email.hasError('email') ? 'Not a valid email' : '';
  }

  getPasswordErrorMessage() {
    return 'You must enter a value';
  }
}
