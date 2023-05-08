import { Component, OnInit } from '@angular/core';
import { ResultService } from './result.service';
import { error } from 'console';

@Component({
  selector: 'app-generate-results',
  templateUrl: './generate-results.component.html',
  styleUrls: ['./generate-results.component.css'],
})
export class GenerateResultsComponent implements OnInit {
  image: any;
  imageLoading: boolean;

  constructor(private service: ResultService) {
    this.imageLoading = true;
  }

  ngOnInit(): void {
    this.service.getBaseImage().subscribe(
      (data) => {
        console.log(data);
        this.createImageFromBlob(data);
        this.imageLoading = false;
      },
      (error) => {
        this.imageLoading = true;
        console.log(error);
      }
    );
  }

  createImageFromBlob(image: any) {
    let reader = new FileReader();
    reader.addEventListener(
      'load',
      () => {
        this.image = reader.result;
      },
      false
    );

    if (image) {
      reader.readAsDataURL(image);
    }
  }
}
